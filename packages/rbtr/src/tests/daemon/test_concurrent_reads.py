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
from collections.abc import Generator
from pathlib import Path

import anyio
import pytest

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import SearchRequest, SearchResponse
from rbtr.daemon.server import DaemonServer
from rbtr.index.models import Chunk, ChunkKind, Edge
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code


def _synthetic_chunk(i: int) -> Chunk:
    """Build a small but realistic chunk for a long write loop."""
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


def _hammer_writes(
    store: IndexStore,
    *,
    repo_path: str,
    stop: threading.Event,
    batch_size: int = 20,
    pause_s: float = 0.01,
) -> None:
    """Continuously insert chunks until *stop* is set.

    Mimics the build worker's write pattern: small batches with a
    brief yield so other threads (and other xdist workers) get
    fair access to the disk and asyncio event loops.
    """
    repo_id = store.register_repo(repo_path)
    i = 0
    while not stop.is_set():
        batch = [_synthetic_chunk(i + n) for n in range(batch_size)]
        store.insert_chunks(batch, repo_id=repo_id)
        i += batch_size
        time.sleep(pause_s)


@pytest.fixture
def disk_store(
    tmp_path: Path,
    daemon_commit: str,
    daemon_chunks: list[Chunk],
    daemon_edges: list[Edge],
) -> IndexStore:
    """Disk-backed store seeded with the same fixtures as `seeded_store`.

    Uses a real DuckDB file so MVCC / WAL contention matches
    production.
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
    """Daemon backed by *disk_store* (real DuckDB file)."""
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


# Generous budget: the assertion is "reads aren't blocked behind writes",
# not "reads are zero-cost".  Under -n auto on a busy CI machine the
# event loop can be starved enough to miss a tight bound; pin loosely
# enough to be robust without losing the regression signal.
_SEARCH_BUDGET_S = 3.0


def test_search_returns_within_budget_during_writes(
    running_server_disk: DaemonServer,
    disk_store: IndexStore,
) -> None:
    """Search must complete quickly even while writes hammer the store.

    Reproduces the production symptom: rbtr-eval saw 'rbtr search'
    fall back to inline mode while the daemon was indexing, because
    the daemon's read handler was blocked behind the write cursor.
    """
    stop = threading.Event()
    writer = threading.Thread(
        target=_hammer_writes,
        args=(disk_store,),
        kwargs={"repo_path": "/test/writer", "stop": stop},
        daemon=True,
    )
    writer.start()
    try:
        time.sleep(0.2)
        with DaemonClient(running_server_disk.sock_dir) as client:
            t0 = time.monotonic()
            resp = client.send(SearchRequest(repo="/test/repo", query="load_config"))
            elapsed = time.monotonic() - t0
    finally:
        stop.set()
        writer.join(timeout=5)

    assert isinstance(resp, SearchResponse)
    assert elapsed < _SEARCH_BUDGET_S, (
        f"search took {elapsed * 1000:.0f} ms while writes were in flight; "
        f"expected < {_SEARCH_BUDGET_S * 1000:.0f} ms.  A long write is "
        "blocking the read handler."
    )


@pytest.mark.parametrize("query", ["load_config", "config", "Application"])
def test_searches_serve_throughout_long_write(
    running_server_disk: DaemonServer,
    disk_store: IndexStore,
    query: str,
) -> None:
    """Multiple searches in sequence all return promptly during writes."""
    stop = threading.Event()
    writer = threading.Thread(
        target=_hammer_writes,
        args=(disk_store,),
        kwargs={"repo_path": "/test/writer2", "stop": stop},
        daemon=True,
    )
    writer.start()
    try:
        time.sleep(0.2)
        with DaemonClient(running_server_disk.sock_dir) as client:
            for _ in range(5):
                t0 = time.monotonic()
                resp = client.send(SearchRequest(repo="/test/repo", query=query))
                elapsed = time.monotonic() - t0
                assert isinstance(resp, SearchResponse)
                assert elapsed < _SEARCH_BUDGET_S, (
                    f"search took {elapsed * 1000:.0f} ms while writes were in flight"
                )
    finally:
        stop.set()
        writer.join(timeout=5)
