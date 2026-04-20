"""Regression tests for daemon concurrency.

Read-side handlers must serve responses promptly even while the
build worker is mid-write.  Uses a disk-backed store so the
MVCC / WAL behaviour matches production; the build is simulated
directly on the store from a thread so the test doesn't have to
drive a full build pipeline.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator, Iterator
from pathlib import Path

import anyio
import pytest

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import SearchRequest, SearchResponse
from rbtr.daemon.server import DaemonServer
from rbtr.index.models import Chunk, ChunkKind, Edge
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code


@pytest.fixture
def search_budget_s() -> float:
    """Per-call search latency budget while writes are in flight.

    Generous so xdist parallelism doesn't false-fail on a busy
    machine; the assertion is `reads aren't blocked behind
    writes`, not `reads are zero-cost`.
    """
    return 3.0


@pytest.fixture
def disk_store(
    tmp_path: Path,
    daemon_commit: str,
    daemon_chunks: list[Chunk],
    daemon_edges: list[Edge],
) -> IndexStore:
    """Disk-backed store seeded with the same fixtures as `seeded_store`.

    Real DuckDB file so MVCC / WAL contention matches production.
    """
    store = IndexStore(tmp_path / "index.duckdb")
    repo_id = store.register_repo("/test/repo")
    store.insert_chunks(daemon_chunks, repo_id=repo_id)
    for c in daemon_chunks:
        store.insert_snapshot(daemon_commit, c.file_path, c.blob_sha, repo_id=repo_id)
    store.insert_edges(daemon_edges, daemon_commit, repo_id=repo_id)
    store.mark_indexed(repo_id, daemon_commit)
    store.rebuild_fts_index()
    return store


@pytest.fixture
def running_server_disk(sock_dir: Path, disk_store: IndexStore) -> Generator[DaemonServer]:
    """Daemon backed by *disk_store*."""
    server = DaemonServer(
        sock_dir, store=disk_store, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    t = threading.Thread(target=lambda: anyio.run(server.serve), daemon=True)
    t.start()
    rpc_path = sock_dir / "daemon.rpc"
    for _ in range(100):
        if rpc_path.exists():
            break
        time.sleep(0.02)
    yield server
    server.request_shutdown()
    t.join(timeout=3)


@pytest.fixture
def background_writer(disk_store: IndexStore) -> Iterator[None]:
    """Hammer *disk_store* with chunk inserts for the duration of the test.

    Runs synthetic 20-chunk batches in a tight loop with a brief
    pause to leave room for the daemon's REP loop, the asyncio
    event loop, and other xdist workers on the same machine.
    Yields after a short warm-up so the contention window is open
    when the test fires its first request; stops the writer thread
    on exit.
    """
    stop = threading.Event()

    def make_chunk(i: int) -> Chunk:
        name = f"slow_fn_{i}"
        content = f"def {name}():\n    return {i}\n"
        return Chunk(
            id=f"slow_{i:08d}",
            blob_sha=f"slowblob_{i:08d}",
            file_path=f"src/slow_{i}.py",
            kind=ChunkKind.FUNCTION,
            name=name,
            content=content,
            content_tokens=tokenise_code(content),
            name_tokens=tokenise_code(name),
            line_start=1,
            line_end=2,
        )

    def hammer() -> None:
        # Register the repo from inside the worker thread so the
        # cursor stays thread-local; sharing a cursor across threads
        # races with the daemon's FTS rebuild path.
        repo_id = disk_store.register_repo("/test/concurrent_writer")
        i = 0
        batch_size = 20
        while not stop.is_set():
            disk_store.insert_chunks(
                [make_chunk(i + n) for n in range(batch_size)], repo_id=repo_id
            )
            i += batch_size
            time.sleep(0.01)

    thread = threading.Thread(target=hammer, daemon=True)
    thread.start()
    try:
        time.sleep(0.2)
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def test_search_returns_within_budget_during_writes(
    running_server_disk: DaemonServer,
    background_writer: None,
    search_budget_s: float,
) -> None:
    """One search returns promptly while writes hammer the store.

    Reproduces the production symptom: rbtr-eval saw `rbtr search`
    fall back to inline mode while the daemon was indexing because
    the daemon's read handler was thought to be blocked behind the
    write cursor.  Closes that hypothesis.
    """
    with DaemonClient(running_server_disk.sock_dir) as client:
        t0 = time.monotonic()
        resp = client.send(SearchRequest(repo="/test/repo", query="load_config"))
        elapsed = time.monotonic() - t0

    assert isinstance(resp, SearchResponse)
    assert elapsed < search_budget_s, (
        f"search took {elapsed * 1000:.0f} ms while writes were in flight; "
        f"expected < {search_budget_s * 1000:.0f} ms.  A long write is "
        "blocking the read handler."
    )


@pytest.mark.parametrize("query", ["load_config", "config", "Application"])
def test_searches_serve_throughout_long_write(
    running_server_disk: DaemonServer,
    background_writer: None,
    search_budget_s: float,
    query: str,
) -> None:
    """Five sequential searches all return promptly during writes."""
    with DaemonClient(running_server_disk.sock_dir) as client:
        for _ in range(5):
            t0 = time.monotonic()
            resp = client.send(SearchRequest(repo="/test/repo", query=query))
            elapsed = time.monotonic() - t0
            assert isinstance(resp, SearchResponse)
            assert elapsed < search_budget_s, (
                f"search took {elapsed * 1000:.0f} ms while writes were in flight"
            )
