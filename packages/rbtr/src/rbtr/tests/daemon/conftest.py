"""Shared fixtures for daemon tests.

All test data is exposed as fixtures.  The `seeded_store`
fixture composes individual chunk/edge fixtures into the root
`store`.  `running_server` and `running_server_with_index`
start a real `DaemonServer` on an IPC socket through
`serve_daemon`, which owns startup and shutdown for every test
that needs a live daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import threading
from collections.abc import Generator, Iterator
from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.server import DaemonServer
from rbtr.domain.models import ChunkKind, Edge, EdgeKind, FileSnapshot, SnapshotRef
from rbtr.index.staging import TokenisedChunk
from rbtr.index.store import IndexStore

from ..index.conftest import make_chunk

# ── Data fixtures ────────────────────────────────────────────────────


@pytest.fixture
def daemon_commit(fake_repo: str) -> str:
    """HEAD SHA of the fake repo — used as the indexed commit."""
    repo = pygit2.Repository(fake_repo)
    return str(repo.head.target)


@pytest.fixture
def daemon_chunks() -> list[TokenisedChunk]:
    """A function, a variable, a class, and an import of the function."""
    return [
        make_chunk(
            "fn_config",
            name="load_config",
            content="def load_config(path):\n    return open(path).read()\n",
            path="src/config.py",
            blob="blob_config",
            kind=ChunkKind.FUNCTION,
        ),
        make_chunk(
            "var_config",
            name="MAX_SIZE",
            content="MAX_SIZE = 100\n",
            path="src/config.py",
            blob="blob_config",
            kind=ChunkKind.VARIABLE,
        ),
        make_chunk(
            "cls_app",
            name="Application",
            content="class Application:\n    pass\n",
            path="src/app.py",
            blob="blob_app",
            kind=ChunkKind.CLASS,
        ),
        make_chunk(
            "imp_config",
            name="from config import load_config",
            content="from config import load_config",
            path="src/app.py",
            blob="blob_app",
            kind=ChunkKind.IMPORT,
        ),
    ]


@pytest.fixture
def daemon_edges(daemon_chunks: list[TokenisedChunk]) -> list[Edge]:
    """The import chunk edges into the function it imports.

    Built from the chunks themselves: an id is derived from content, so
    an edge written with a literal would point at nothing.
    """
    by_name = {c.name: c for c in daemon_chunks}
    imp = by_name["from config import load_config"]
    fn = by_name["load_config"]
    return [
        Edge(
            source_id=imp.id,
            target_id=fn.id,
            kind=EdgeKind.IMPORTS,
            source_path=imp.file_path,
            target_path=fn.file_path,
        ),
    ]


# ── Server-support fixtures ─────────────────────────────────────────


@pytest.fixture
def runtime_dir() -> Path:
    """Short temp dir for IPC sockets (avoids AF_UNIX path limit)."""
    return Path(tempfile.mkdtemp(prefix="rbtr"))


@pytest.fixture
def unindexed_store(fake_repo: str, store: IndexStore) -> IndexStore:
    """Real repo registered, HEAD not indexed (a stale watched ref)."""
    with store.session() as ws:
        ws.register_repo(fake_repo)
    return store


@pytest.fixture
def seeded_store(
    fake_repo: str,
    daemon_commit: str,
    daemon_chunks: list[TokenisedChunk],
    daemon_edges: list[Edge],
    store: IndexStore,
) -> IndexStore:
    """The root store pre-loaded with daemon test data for one repo."""
    chunks = daemon_chunks
    with store.session() as ws:
        repo_id = ws.register_repo(fake_repo)
        for c in chunks:
            ws.add_chunk(c)
        ws.insert_snapshots(
            [
                FileSnapshot(snapshot_sha=daemon_commit, file_path=c.file_path, blob_sha=c.blob_sha)
                for c in chunks
            ],
            repo_id=repo_id,
        )
        ws.insert_edges(daemon_edges, at=SnapshotRef(repo_id=repo_id, snapshot_sha=daemon_commit))
        ws.mark_indexed(at=SnapshotRef(repo_id=repo_id, snapshot_sha=daemon_commit))
    return store


@pytest.fixture
def changed_head(seeded_store: IndexStore, fake_repo: str) -> str:
    """Add a second indexed commit to `seeded_store`; return its SHA.

    Diffs against the base commit (`daemon_commit`): `load_config` is
    modified and a new `helper` is added. The head SHA is synthetic —
    the diff is keyed on the stored snapshots, not on git tree content.
    """
    head = "f" * 40
    repo_id = seeded_store.resolve_repo(fake_repo)
    with seeded_store.session() as ws:
        ws.add_chunk(
            make_chunk(
                "fn_config_v2",
                name="load_config",
                content="def load_config(path):\n    with open(path) as f:\n        return f.read()\n",
                path="src/config.py",
                blob="blob_config_v2",
                kind=ChunkKind.FUNCTION,
            )
        )
        ws.add_chunk(
            make_chunk(
                "fn_helper",
                name="helper",
                content="def helper():\n    return 1\n",
                path="src/config.py",
                blob="blob_config_v2",
                kind=ChunkKind.FUNCTION,
            )
        )
        ws.insert_snapshots(
            [
                FileSnapshot(
                    snapshot_sha=head, file_path="src/config.py", blob_sha="blob_config_v2"
                ),
                FileSnapshot(snapshot_sha=head, file_path="src/app.py", blob_sha="blob_app"),
            ],
            repo_id=repo_id,
        )
        ws.mark_indexed(at=SnapshotRef(repo_id=repo_id, snapshot_sha=head))
    return head


@contextlib.contextmanager
def serving(server: DaemonServer) -> Iterator[DaemonServer]:
    """Serve *server* on a thread for the duration of the block.

    Every daemon fixture goes through here, so starting one is a single
    construct rather than four statements repeated per module.

    The exit check is the point: `join` with a timeout returns whether
    or not the daemon stopped, so an unasserted join lets a hung daemon
    leak a thread — still holding its store and IPC sockets — into every
    test that follows.

    The wait is long because `serve()` finishes the job in flight before
    it returns: a test that starts a real build legitimately takes
    seconds to stop, and more under xdist.  What is being caught is a
    daemon that never stops, not a slow one.
    """
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    assert server.wait_ready(), "daemon did not start within timeout"
    try:
        yield server
    finally:
        server.request_shutdown()
        thread.join(timeout=30)
        assert not thread.is_alive(), "daemon thread still running after shutdown"


@pytest.fixture
def running_daemon(
    runtime_dir: Path,
    seeded_store: IndexStore,
    stub_embedding_model: None,
) -> Generator[DaemonServer]:
    """A served daemon over a seeded index.

    Override in a module that needs a different one — a bespoke store,
    embedder or poll interval — building the server and wrapping it in
    `serving` the same way.

    Uses the stub embedder: these tests exercise routing / read /
    status, not vector quality, so the daemon must not load the
    real GGUF (and never touch the GPU).
    """
    with serving(
        DaemonServer(
            runtime_dir, store=seeded_store, idle_poll_interval=60.0, busy_poll_interval=60.0
        )
    ) as server:
        yield server
