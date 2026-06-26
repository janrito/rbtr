"""Shared fixtures for daemon tests.

All test data is exposed as fixtures.  The `seeded_store`
fixture composes individual chunk/edge fixtures into an
`IndexStore`.  `running_server` and `running_server_with_index`
start a real `DaemonServer` on an IPC socket and yield until the
test finishes.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.server import DaemonServer
from rbtr.index.models import ChunkKind, Edge, EdgeKind, Snapshot
from rbtr.index.store import IndexStore

from ..index.conftest import make_chunk

# ── Data fixtures ────────────────────────────────────────────────────


@pytest.fixture
def daemon_commit(fake_repo: str) -> str:
    """HEAD SHA of the fake repo — used as the indexed commit."""
    repo = pygit2.Repository(fake_repo)
    return str(repo.head.target)


@pytest.fixture
def daemon_edges() -> list[Edge]:
    return [
        Edge(source_id="imp_config", target_id="fn_config", kind=EdgeKind.IMPORTS),
    ]


# ── Server-support fixtures ─────────────────────────────────────────


@pytest.fixture
def runtime_dir() -> Path:
    """Short temp dir for IPC sockets (avoids AF_UNIX path limit)."""
    return Path(tempfile.mkdtemp(prefix="rbtr"))


@pytest.fixture
def unindexed_store(fake_repo: str) -> Generator[IndexStore]:
    """Real repo registered, HEAD not indexed (a stale watched ref)."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        ws.register_repo(fake_repo)
    yield store
    store.close()


@pytest.fixture
def seeded_store(
    fake_repo: str,
    daemon_commit: str,
    daemon_edges: list[Edge],
) -> Generator[IndexStore]:
    """In-memory store pre-loaded with daemon test data for one repo."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(fake_repo)
        # A function, a class, and an import (the import edges into
        # the function — see daemon_edges).
        chunks = [
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
        for c in chunks:
            ws.add_chunk(c)
        ws.insert_snapshots(
            [
                Snapshot(commit_sha=daemon_commit, file_path=c.file_path, blob_sha=c.blob_sha)
                for c in chunks
            ],
            repo_id=repo_id,
        )
        ws.insert_edges(daemon_edges, daemon_commit, repo_id=repo_id)
        ws.mark_indexed(repo_id, daemon_commit)
    yield store
    store.close()


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
                Snapshot(commit_sha=head, file_path="src/config.py", blob_sha="blob_config_v2"),
                Snapshot(commit_sha=head, file_path="src/app.py", blob_sha="blob_app"),
            ],
            repo_id=repo_id,
        )
        ws.mark_indexed(repo_id, head)
    return head


@pytest.fixture
def running_server(runtime_dir: Path) -> Generator[DaemonServer]:
    """Server with no index — for ping/shutdown tests."""
    server = DaemonServer(runtime_dir, store=None, idle_poll_interval=60.0, busy_poll_interval=60.0)
    t = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    t.start()
    assert server.wait_ready(), "daemon did not start within timeout"
    yield server
    server.request_shutdown()
    t.join(timeout=3)


@pytest.fixture
def running_server_with_index(
    runtime_dir: Path, seeded_store: IndexStore, stub_embedding_model: None
) -> Generator[DaemonServer]:
    """Server with a seeded index — for handler tests.

    Uses the stub embedder: these tests exercise routing / read /
    status, not vector quality, so the daemon must not load the
    real GGUF (and never touch the GPU).
    """
    server = DaemonServer(
        runtime_dir, store=seeded_store, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    t = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    t.start()
    assert server.wait_ready(), "daemon did not start within timeout"
    yield server
    server.request_shutdown()
    t.join(timeout=3)


@dataclass(frozen=True)
class TwoRepoServer:
    """A running daemon over two indexed repos sharing a symbol."""

    server: DaemonServer
    path_a: str
    path_b: str
    shared_name: str


def _init_bare_commit_repo(path: Path) -> tuple[str, str]:
    """Create a one-commit git repo; return `(workdir, head_sha)`."""
    repo = pygit2.init_repository(str(path), bare=False, initial_head="main")
    sig = pygit2.Signature("t", "t@t.t")
    head = repo.create_commit("refs/heads/main", sig, sig, "init", repo.TreeBuilder().write(), [])
    return str(path), str(head)


@pytest.fixture
def two_repo_server(
    runtime_dir: Path, tmp_path: Path, stub_embedding_model: None
) -> Generator[TwoRepoServer]:
    """Daemon over an in-memory store with two indexed git repos.

    Uses the stub embedder (cross-repo routing, not vector quality),
    so the daemon never loads the real GGUF or touches the GPU.

    Each repo holds a uniquely-named chunk plus `shared_fn` (same
    name in both), so cross-repo merge, workspace isolation, and
    non-search isolation are all exercisable through the socket.
    Real git repos (the handler resolves HEAD); chunks seeded under
    each repo's real id and HEAD sha.
    """
    path_a, head_a = _init_bare_commit_repo(tmp_path / "repo_a")
    path_b, head_b = _init_bare_commit_repo(tmp_path / "repo_b")

    store = IndexStore(writable=True)
    with store.session() as ws:
        id_a = ws.register_repo(path_a)
        id_b = ws.register_repo(path_b)
        for repo_id, uniq, head in ((id_a, "alpha", head_a), (id_b, "beta", head_b)):
            ws.add_chunk(make_chunk(f"{uniq}_id", name=f"{uniq}_fn", path=f"{uniq}.py"))
            ws.add_chunk(make_chunk(f"shared_{uniq}", name="shared_fn", path="shared.py"))
            ws.insert_snapshots(
                [
                    Snapshot(commit_sha=head, file_path=f"{uniq}.py", blob_sha=f"blob_{uniq}_id"),
                    Snapshot(
                        commit_sha=head, file_path="shared.py", blob_sha=f"blob_shared_{uniq}"
                    ),
                ],
                repo_id=repo_id,
            )
            ws.mark_indexed(repo_id, head)

    server = DaemonServer(
        runtime_dir, store=store, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    t = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    t.start()
    assert server.wait_ready(), "daemon did not start within timeout"
    yield TwoRepoServer(server=server, path_a=path_a, path_b=path_b, shared_name="shared_fn")
    server.request_shutdown()
    t.join(timeout=3)
    store.close()
