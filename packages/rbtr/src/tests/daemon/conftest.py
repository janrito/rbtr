"""Shared fixtures for daemon tests.

All test data is exposed as fixtures.  The ``seeded_store``
fixture composes individual chunk/edge fixtures into an
``IndexStore``.  ``running_server`` and ``running_server_with_index``
start a real ``DaemonServer`` on an IPC socket and yield until the
test finishes.
"""

from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Generator
from pathlib import Path

import anyio
import pytest

from rbtr.daemon.server import DaemonServer
from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code


# ── Data fixtures ────────────────────────────────────────────────────


@pytest.fixture
def daemon_commit() -> str:
    return "HEAD"


@pytest.fixture
def daemon_func_chunk() -> Chunk:
    content = "def load_config(path):\n    return open(path).read()\n"
    name = "load_config"
    return Chunk(
        id="fn_config",
        blob_sha="blob_config",
        file_path="src/config.py",
        kind=ChunkKind.FUNCTION,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=2,
    )


@pytest.fixture
def daemon_class_chunk() -> Chunk:
    content = "class Application:\n    pass\n"
    name = "Application"
    return Chunk(
        id="cls_app",
        blob_sha="blob_app",
        file_path="src/app.py",
        kind=ChunkKind.CLASS,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=2,
    )


@pytest.fixture
def daemon_import_chunk() -> Chunk:
    content = "from config import load_config"
    name = "from config import load_config"
    return Chunk(
        id="imp_config",
        blob_sha="blob_app",
        file_path="src/app.py",
        kind=ChunkKind.IMPORT,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=5,
        line_end=5,
    )


@pytest.fixture
def daemon_chunks(
    daemon_func_chunk: Chunk,
    daemon_class_chunk: Chunk,
    daemon_import_chunk: Chunk,
) -> list[Chunk]:
    return [daemon_func_chunk, daemon_class_chunk, daemon_import_chunk]


@pytest.fixture
def daemon_edges() -> list[Edge]:
    return [
        Edge(source_id="imp_config", target_id="fn_config", kind=EdgeKind.IMPORTS),
    ]


# ── Server-support fixtures ─────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out the embedding model — no GGUF loading in daemon tests."""
    monkeypatch.setattr(
        "rbtr.index.store.IndexStore.search_by_text",
        lambda *_args, **_kwargs: [],
    )


@pytest.fixture
def sock_dir() -> Path:
    """Short temp dir for IPC sockets (avoids AF_UNIX path limit)."""
    return Path(tempfile.mkdtemp(prefix="rbtr"))


@pytest.fixture
def seeded_store(
    daemon_commit: str,
    daemon_chunks: list[Chunk],
    daemon_edges: list[Edge],
) -> IndexStore:
    """In-memory store pre-loaded with daemon test data for one repo."""
    store = IndexStore()
    repo_id = store.register_repo("/test/repo")
    store.insert_chunks(daemon_chunks, repo_id=repo_id)
    for c in daemon_chunks:
        store.insert_snapshot(
            daemon_commit, c.file_path, c.blob_sha, repo_id=repo_id
        )
    store.insert_edges(daemon_edges, daemon_commit, repo_id=repo_id)
    store.mark_indexed(repo_id, daemon_commit)
    store.rebuild_fts_index()
    return store


@pytest.fixture
def running_server(sock_dir: Path) -> Generator[DaemonServer]:
    """Server with no index — for ping/shutdown tests."""
    server = DaemonServer(sock_dir, store=None, idle_poll_interval=60.0, busy_poll_interval=60.0)
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
def running_server_with_index(
    sock_dir: Path, seeded_store: IndexStore
) -> Generator[DaemonServer]:
    """Server with a seeded index — for handler tests."""
    server = DaemonServer(sock_dir, store=seeded_store, idle_poll_interval=60.0, busy_poll_interval=60.0)
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
