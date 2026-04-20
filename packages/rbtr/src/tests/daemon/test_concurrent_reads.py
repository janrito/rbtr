"""Regression tests for daemon concurrency.

Drives a real `BuildIndexRequest` through ZMQ, waits for the
build worker to start, then fires `SearchRequest`s against the
same live daemon.  Exercises the full stack: socket dispatch,
build queue, worker thread, tree-sitter extraction, DuckDB
inserts, and the read-side handlers.  The narrower previous
version (writer thread bypassing the build pipeline) let a
regression in the real build-vs-search contention slip through
because it never exercised that code.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from pathlib import Path

import anyio
import pygit2
import pytest

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    BuildIndexRequest,
    OkResponse,
    SearchRequest,
    SearchResponse,
    StatusRequest,
    StatusResponse,
)
from rbtr.daemon.server import DaemonServer
from rbtr.index.store import IndexStore


@pytest.fixture
def search_budget_s() -> float:
    """Per-call search latency budget during a live build.

    Generous so xdist parallelism doesn't false-fail on a busy
    machine.  The assertion is `reads aren't blocked behind the
    build`, not `reads are zero-cost`.  A true serialisation
    would show latencies on the order of the full build (tens
    of seconds for 100 files with embeddings); 10s comfortably
    rules that out while leaving room for first-run model loads
    and GIL contention under xdist.
    """
    return 10.0


@pytest.fixture
def large_repo(tmp_path: Path) -> tuple[Path, str]:
    """Git repo with enough files that indexing takes seconds.

    Returns `(repo_path, head_sha)`.  One commit; ~100 Python
    files each carrying a class, a method, and a top-level
    function with docstrings so every standard chunk kind is
    exercised by the subsequent indexing run.
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo = pygit2.init_repository(str(repo_path), bare=False)
    index = repo.index
    for i in range(100):
        content = (
            f'"""module {i}."""\n\n'
            f"class Thing{i}:\n"
            f'    """Docstring {i}."""\n'
            f"    def method_{i}(self, x):\n"
            f"        return x + {i}\n\n"
            f"def helper_{i}(a, b):\n"
            f'    """Top-level helper {i}."""\n'
            f"    return a + b + {i}\n"
        )
        rel = f"src/mod_{i}.py"
        full = repo_path / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        index.add(rel)
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    sha = str(repo.create_commit("HEAD", sig, sig, "init", tree_oid, []))
    return repo_path, sha


@pytest.fixture
def daemon_store(tmp_path: Path) -> IndexStore:
    """Disk-backed, empty store for the daemon."""
    return IndexStore(tmp_path / "index.duckdb")


@pytest.fixture
def running_daemon(sock_dir: Path, daemon_store: IndexStore) -> Generator[DaemonServer]:
    """A real daemon serving *daemon_store*.

    `sock_dir` comes from `tests/daemon/conftest.py` and lives
    under a short `/tmp/rbtr*` path (macOS AF_UNIX has a
    103-char limit; `tmp_path` under xdist would exceed it).
    """
    server = DaemonServer(
        sock_dir,
        store=daemon_store,
        idle_poll_interval=60.0,
        busy_poll_interval=60.0,
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


def _wait_for_build_start(client: DaemonClient, repo_path: Path, deadline_s: float) -> None:
    """Poll `rbtr status` until the build worker has a repo active.

    Returns when `active_job` is set *or* the index exists;
    raises `TimeoutError` if neither happens within *deadline_s*.
    """
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        resp = client.send(StatusRequest(repo=str(repo_path)))
        if isinstance(resp, StatusResponse) and resp.active_job is not None:
            return
        time.sleep(0.05)
    msg = "build worker never became active"
    raise TimeoutError(msg)


def test_search_returns_promptly_during_live_build(
    running_daemon: DaemonServer,
    large_repo: tuple[Path, str],
    search_budget_s: float,
) -> None:
    """Drive a full BuildIndexRequest; fire SearchRequests during the build.

    The daemon must keep serving read RPCs while the build worker
    parses, inserts, and writes to DuckDB on its own thread.
    Regression guard: if anything (ZMQ dispatch, cursor sharing,
    DuckDB locking) ever serialises reads behind the build, this
    fails.
    """
    repo_path, _sha = large_repo

    with DaemonClient(running_daemon.sock_dir) as client:
        build_resp = client.send(BuildIndexRequest(repo=str(repo_path)))
        assert isinstance(build_resp, OkResponse)

        _wait_for_build_start(client, repo_path, deadline_s=5.0)

        latencies: list[float] = []
        for _ in range(10):
            t0 = time.monotonic()
            search_resp = client.send(
                SearchRequest(repo=str(repo_path), query="helper_42", limit=5)
            )
            elapsed = time.monotonic() - t0
            assert isinstance(search_resp, SearchResponse)
            latencies.append(elapsed)
            time.sleep(0.05)

    assert max(latencies) < search_budget_s, (
        f"slowest search during build took {max(latencies) * 1000:.0f} ms; "
        f"expected < {search_budget_s * 1000:.0f} ms.  Reads are being "
        f"blocked behind the build.  All latencies (ms): "
        f"{[int(x * 1000) for x in latencies]}"
    )


def test_status_returns_promptly_during_live_build(
    running_daemon: DaemonServer,
    large_repo: tuple[Path, str],
    search_budget_s: float,
) -> None:
    """Status must also serve while the build is in flight.

    Covers the same concurrency property for the status path
    (which has its own code path through `handle_status` ->
    `_build_queue.snapshot_status()`).
    """
    repo_path, _sha = large_repo

    with DaemonClient(running_daemon.sock_dir) as client:
        build_resp = client.send(BuildIndexRequest(repo=str(repo_path)))
        assert isinstance(build_resp, OkResponse)

        _wait_for_build_start(client, repo_path, deadline_s=5.0)

        for _ in range(10):
            t0 = time.monotonic()
            resp = client.send(StatusRequest(repo=str(repo_path)))
            elapsed = time.monotonic() - t0
            assert isinstance(resp, StatusResponse)
            assert elapsed < search_budget_s, f"status took {elapsed * 1000:.0f} ms during build"
            time.sleep(0.05)
