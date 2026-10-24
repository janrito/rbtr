"""Shared fixtures for daemon tests.

All test data is exposed as fixtures.  The `seeded_index`
fixture composes individual chunk/edge fixtures into the root
`index`.  `running_server` and `running_server_with_index`
start a real `DaemonServer` on an IPC socket through
`serve_daemon`, which owns startup and shutdown for every test
that needs a live daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import threading
from collections.abc import Callable, Generator, Iterator
from pathlib import Path

import pygit2
import pytest
import zmq
from pytest_mock import MockerFixture

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import Request, Response
from rbtr.daemon.server import DaemonServer
from rbtr.domain.models import ChunkKind, Edge, EdgeKind, FileSnapshot, Repo
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index
from rbtr.index.staging import TokenisedChunk

from ..index.conftest import make_chunk


@pytest.fixture
def rpc_peer(runtime_dir: Path) -> Generator[zmq.Socket[bytes]]:
    with zmq.Context() as context, context.socket(zmq.DEALER) as peer:
        peer.setsockopt(zmq.LINGER, 0)
        peer.setsockopt(zmq.RCVTIMEO, 5000)
        peer.connect(f"ipc://{runtime_dir / 'daemon.rpc'}")
        yield peer


@pytest.fixture
def request_peer(runtime_dir: Path) -> Generator[zmq.Socket[bytes]]:
    with zmq.Context() as context, context.socket(zmq.REQ) as peer:
        peer.setsockopt(zmq.LINGER, 0)
        peer.setsockopt(zmq.RCVTIMEO, 5000)
        peer.connect(f"ipc://{runtime_dir / 'daemon.rpc'}")
        yield peer


@pytest.fixture
def correlated_request_peer(request_peer: zmq.Socket[bytes]) -> zmq.Socket[bytes]:
    request_peer.setsockopt(zmq.REQ_CORRELATE, 1)
    return request_peer


@pytest.fixture
def rpc_send(runtime_dir: Path) -> Callable[[Request], Response]:
    def send(message: Request) -> Response:
        with DaemonClient(runtime_dir, wait_budget_s=15) as client:
            return client.send(message)

    return send


@pytest.fixture
def query_started() -> threading.Event:
    return threading.Event()


@pytest.fixture
def query_released() -> Generator[threading.Event]:
    released = threading.Event()
    yield released
    released.set()


@pytest.fixture
def held_query(
    query_started: threading.Event,
    query_released: threading.Event,
    stub_embedding_model: None,
    mocker: MockerFixture,
) -> None:
    original = Embedder.embed_single

    def embed_single(embedder: Embedder, text: str) -> list[float]:
        query_started.set()
        if not query_released.wait(timeout=20):
            msg = "query was not released"
            raise TimeoutError(msg)
        return original(embedder, text)

    mocker.patch.object(Embedder, "embed_single", embed_single)


@pytest.fixture
def write_started() -> threading.Event:
    return threading.Event()


@pytest.fixture
def write_released() -> Generator[threading.Event]:
    released = threading.Event()
    yield released
    released.set()


@pytest.fixture
def held_watch(
    seeded_index: Index,
    write_started: threading.Event,
    write_released: threading.Event,
    mocker: MockerFixture,
) -> None:
    original = seeded_index.register

    def register(path: str) -> Repo:
        write_started.set()
        if not write_released.wait(20):
            msg = "watch write was not released"
            raise TimeoutError(msg)
        return original(path)

    mocker.patch.object(seeded_index, "register", side_effect=register)


@pytest.fixture
def fake_repo_subdir(fake_repo: str) -> str:
    """A directory inside `fake_repo`."""
    path = Path(fake_repo) / "sub"
    path.mkdir()
    return str(path)


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


@pytest.fixture
def runtime_dir() -> Path:
    """Short temp dir for IPC sockets (avoids AF_UNIX path limit)."""
    return Path(tempfile.mkdtemp(prefix="rbtr"))


@pytest.fixture
def registered_index(fake_repo: str, index: Index) -> Index:
    """Real repo registered, HEAD not indexed (a stale watched ref)."""
    index.register(fake_repo)
    return index


@pytest.fixture
def empty_index() -> Generator[Index]:
    """An in-memory writable `Index` no other fixture fills.

    A test fed by cases sets up every case's fixtures for each case, and
    `seeded_index` fills `index`; a case that needs an empty index asks
    for this one.
    """
    index = Index(writable=True)
    yield index
    index.close()


@pytest.fixture
def seeded_index(
    fake_repo: str,
    daemon_commit: str,
    daemon_chunks: list[TokenisedChunk],
    daemon_edges: list[Edge],
    index: Index,
) -> Index:
    """The root store pre-loaded with daemon test data for one repo."""
    chunks = daemon_chunks
    repo = index.register(fake_repo)
    with index.session(repo) as ws:
        for c in chunks:
            ws.add_chunk(c)
        ws.insert_snapshots(
            [
                FileSnapshot(snapshot_sha=daemon_commit, file_path=c.file_path, blob_sha=c.blob_sha)
                for c in chunks
            ],
        )
        ws.insert_edges(daemon_edges, snapshot_sha=daemon_commit)
        ws.mark_indexed(snapshot_sha=daemon_commit)
    return index


@pytest.fixture
def changed_head(seeded_index: Index, fake_repo: str) -> str:
    """Add a second indexed commit to `seeded_index`; return its SHA.

    Diffs against the base commit (`daemon_commit`): `load_config` is
    modified and a new `helper` is added. The head SHA is synthetic —
    the diff is keyed on the stored snapshots, not on git tree content.
    """
    head = "f" * 40
    repo = seeded_index.repo(fake_repo)
    with seeded_index.session(repo) as ws:
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
        )
        ws.mark_indexed(snapshot_sha=head)
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
    seeded_index: Index,
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
            runtime_dir, index=seeded_index, idle_poll_interval=60.0, busy_poll_interval=60.0
        )
    ) as server:
        yield server
