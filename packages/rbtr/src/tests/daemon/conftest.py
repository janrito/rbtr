"""Shared fixtures for daemon tests."""

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

COMMIT = "HEAD"


# ── Test data ────────────────────────────────────────────────────────

FUNC_CHUNK = Chunk(
    id="fn_config",
    blob_sha="blob_config",
    file_path="src/config.py",
    kind=ChunkKind.FUNCTION,
    name="load_config",
    content="def load_config(path):\n    return open(path).read()\n",
    line_start=1,
    line_end=2,
)

CLASS_CHUNK = Chunk(
    id="cls_app",
    blob_sha="blob_app",
    file_path="src/app.py",
    kind=ChunkKind.CLASS,
    name="Application",
    content="class Application:\n    pass\n",
    line_start=1,
    line_end=2,
)

IMPORT_CHUNK = Chunk(
    id="imp_config",
    blob_sha="blob_app",
    file_path="src/app.py",
    kind=ChunkKind.IMPORT,
    name="from config import load_config",
    content="from config import load_config",
    line_start=5,
    line_end=5,
)

TEST_CHUNKS = [FUNC_CHUNK, CLASS_CHUNK, IMPORT_CHUNK]

TEST_EDGES = [
    Edge(source_id="imp_config", target_id="fn_config", kind=EdgeKind.IMPORTS),
]


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out embedding model — no GGUF loading in daemon tests."""
    monkeypatch.setattr(
        "rbtr.index.store.IndexStore.search_by_text",
        lambda *_args, **_kwargs: [],
    )


@pytest.fixture
def sock_dir() -> Path:
    """Short temp dir for IPC sockets (avoids AF_UNIX path limit)."""
    return Path(tempfile.mkdtemp(prefix="rbtr"))


@pytest.fixture
def seeded_store() -> IndexStore:
    """In-memory store pre-loaded with test data for one repo."""
    store = IndexStore()
    repo_id = store.register_repo("/test/repo")

    for c in TEST_CHUNKS:
        c.content_tokens = tokenise_code(c.content)
        c.name_tokens = tokenise_code(c.name)

    store.insert_chunks(TEST_CHUNKS, repo_id=repo_id)
    for c in TEST_CHUNKS:
        store.insert_snapshot(COMMIT, c.file_path, c.blob_sha, repo_id=repo_id)
    store.insert_edges(TEST_EDGES, COMMIT, repo_id=repo_id)
    store.rebuild_fts_index()
    return store


def start_server(
    sock_dir: Path,
    store: IndexStore | None = None,
    *,
    poll_interval: float = 60.0,
) -> tuple[DaemonServer, threading.Thread]:
    """Start a server and wait for the socket file to appear."""
    server = DaemonServer(sock_dir, store=store, poll_interval=poll_interval)
    t = threading.Thread(target=lambda: anyio.run(server.serve), daemon=True)
    t.start()
    rpc_path = sock_dir / "daemon.rpc"
    for _ in range(100):
        if rpc_path.exists():
            break
        time.sleep(0.02)
    return server, t


@pytest.fixture
def running_server(sock_dir: Path) -> Generator[DaemonServer]:
    """Server with no index — for ping/shutdown tests."""
    server, t = start_server(sock_dir)
    yield server
    server.request_shutdown()
    t.join(timeout=3)


@pytest.fixture
def running_server_with_index(sock_dir: Path, seeded_store: IndexStore) -> Generator[DaemonServer]:
    """Server with a seeded index — for handler tests."""
    server, t = start_server(sock_dir, store=seeded_store)
    yield server
    server.request_shutdown()
    t.join(timeout=3)
