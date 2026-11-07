"""Shared fixtures for daemon tests.

All test data is exposed as fixtures.  The `seeded_index`
fixture composes individual chunk/edge fixtures into the root
`index`.  `running_server` and `running_server_with_index`
start a real `Daemon` on an IPC socket through
`serve_daemon`, which owns startup and shutdown for every test
that needs a live daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import threading
import time
from collections.abc import Callable, Generator, Iterator
from pathlib import Path

import pygit2
import pytest
import zmq
from pytest_mock import MockerFixture

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    GcEndedNotification,
    GcRequest,
    GcResponse,
    Request,
    Response,
    notification_adapter,
)
from rbtr.daemon.server import Daemon
from rbtr.domain.models import GcMode, IndexResult, Repo
from rbtr.index.build import build_index
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index
from rbtr.index.progress import ProgressCallback, RepoProgressCallback

from ..conftest import make_commit


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
def gc_started() -> threading.Event:
    return threading.Event()


@pytest.fixture
def gc_released() -> Generator[threading.Event]:
    released = threading.Event()
    yield released
    released.set()


@pytest.fixture
def held_gc(
    fake_repo: str,
    gc_started: threading.Event,
    gc_released: threading.Event,
    mocker: MockerFixture,
) -> None:
    """A gc request reports one repo of two collected, then waits for `gc_released`."""

    def collect(*_args: object, on_progress: RepoProgressCallback, **_kwargs: object) -> GcResponse:
        on_progress(fake_repo, 1, 2)
        gc_started.set()
        if not gc_released.wait(20):
            msg = "gc was not released"
            raise TimeoutError(msg)
        return GcResponse(
            snapshots_dropped=0,
            file_snapshots_dropped=0,
            edges_dropped=0,
            chunks_freed=0,
            elapsed_seconds=0.0,
            dry_run=True,
        )

    mocker.patch("rbtr.daemon.server.handle_gc_all", collect)


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
def held_build(
    second_repo: str,
    write_started: threading.Event,
    write_released: threading.Event,
    mocker: MockerFixture,
) -> None:
    def build(
        repo_path: str,
        snapshot_sha: str,
        index: Index,
        *,
        on_progress: ProgressCallback,
        stop: threading.Event,
    ) -> IndexResult:
        def report(phase: str, done: int, total: int) -> None:
            on_progress(phase, done, total)
            if repo_path == second_repo and phase == "edges":
                write_started.set()
                assert write_released.wait(20), "build was not released"

        return build_index(repo_path, snapshot_sha, index, on_progress=report, stop=stop)

    mocker.patch("rbtr.index.indexer.build_index", side_effect=build)


@pytest.fixture
def fake_repo_subdir(fake_repo: str) -> str:
    """A directory inside `fake_repo`."""
    path = Path(fake_repo) / "sub"
    path.mkdir()
    return str(path)


@pytest.fixture
def daemon_commit(fake_repo: str) -> str:
    """A commit on `fake_repo`'s `main`, now its HEAD: a config module and an app importing it.

    `src/config.py` holds a variable and a function, `src/app.py` a
    class and an import of the function, so a build gives one chunk of
    each kind and an `imports` edge.
    """
    config_py = b"""\
MAX_SIZE = 100


def load_config(path):
    return open(path).read()
"""
    app_py = b"""\
from config import load_config


class Application:
    pass
"""
    repo = pygit2.Repository(fake_repo)
    files = {"src/config.py": config_py, "src/app.py": app_py}
    sha = str(make_commit(repo, files, parents=[repo.head.peel(pygit2.Commit).id]))
    # The files on disk match HEAD, as after a real commit, so the worktree is clean.
    repo.checkout_head(strategy=pygit2.GIT_CHECKOUT_FORCE)  # type: ignore[no-untyped-call]  # pygit2 untyped
    return sha


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
def seeded_index(fake_repo: str, daemon_commit: str, index: Index) -> Index:
    """The root store with `fake_repo` built at `daemon_commit`."""
    build_index(fake_repo, daemon_commit, index)
    return index


@pytest.fixture
def changed_head(seeded_index: Index, fake_repo: str, daemon_commit: str) -> str:
    """A second commit after `daemon_commit`, built into `seeded_index`; returns its SHA.

    `load_config` is modified and a new `helper` is added. The commit is
    on a branch of its own, so `fake_repo`'s HEAD stays `daemon_commit`.
    """
    config_py = b"""\
MAX_SIZE = 100


def load_config(path):
    with open(path) as f:
        return f.read()


def helper():
    return 1
"""
    app_py = b"""\
from config import load_config


class Application:
    pass
"""
    repo = pygit2.Repository(fake_repo)
    files = {"src/config.py": config_py, "src/app.py": app_py}
    parents = [pygit2.Oid(hex=daemon_commit)]
    sha = str(make_commit(repo, files, parents=parents, ref="refs/heads/changed"))
    build_index(fake_repo, sha, seeded_index)
    return sha


@contextlib.contextmanager
def serving(server: Daemon) -> Iterator[Daemon]:
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


@contextlib.contextmanager
def subscribed(server: Daemon, repo_path: str) -> Iterator[zmq.Socket[bytes]]:
    """A SUB socket on *server*'s notifications that is known to receive them.

    A PUB socket drops what it publishes before a subscription reaches it,
    and the daemon has no handshake to say when one has.  So this sends a
    dry-run gc of *repo_path*, which publishes its progress and `gc_ended`
    and writes nothing, until that `gc_ended` arrives.  The probe's own
    notifications are read here, so the block sees only what it triggers.
    """
    sub: zmq.Socket[bytes] = zmq.Context.instance().socket(zmq.SUB)
    sub.connect(server.pub_addr)
    sub.subscribe(b"")
    try:
        deadline = time.monotonic() + 30
        probe = GcRequest(repo_path=repo_path, mode=GcMode.WATCHED, dry_run=True)
        received = False
        with DaemonClient(server.runtime_dir) as client:
            while not received:
                assert time.monotonic() < deadline, "the subscription received nothing"
                client.send(probe)
                wait_until = time.monotonic() + 2
                while not received and (remaining := wait_until - time.monotonic()) > 0:
                    if not sub.poll(int(remaining * 1000)):
                        break
                    notification = notification_adapter.validate_json(sub.recv())
                    received = isinstance(notification, GcEndedNotification)
        yield sub
    finally:
        sub.close()


@pytest.fixture
def running_daemon(
    runtime_dir: Path,
    seeded_index: Index,
    stub_embedding_model: None,
) -> Generator[Daemon]:
    """A served daemon over a seeded index.

    Override in a module that needs a different one — a bespoke store,
    embedder or poll interval — building the server and wrapping it in
    `serving` the same way.

    Uses the stub embedder: these tests exercise routing / read /
    status, not vector quality, so the daemon must not load the
    real GGUF (and never touch the GPU).
    """
    with serving(
        Daemon(runtime_dir, index=seeded_index, idle_poll_interval=60.0, busy_poll_interval=60.0)
    ) as server:
        yield server
