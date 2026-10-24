"""Tests for how the embed worker shares the daemon with everything else.

Search and embed contend for the GPU and take turns on `_gpu_lock`.
The real `Embedder` runs here with a `StubModel` injected via
`model_loader`, and the stub counts concurrent calls to `embed()` —
the call site where `llama_cpp` would crash in production.

A build wants the job worker instead, which runs one job at a time, so
a build starts only once the embed job returns.  Writes take a third
turn on `_write_sem`.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import MethodType

import pygit2
import pytest
import structlog
import zmq
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    DaemonConfigRequest,
    EmbedEndedNotification,
    GcRequest,
    Notification,
    Response,
    SearchRequest,
    SearchResponse,
    WatchRequest,
    notification_adapter,
    response_adapter,
)
from rbtr.daemon.server import Daemon
from rbtr.domain.models import ChunkKind, FileSnapshot, GcCounts, GcMode, SnapshotRef
from rbtr.domain.tokenise import tokenise_code
from rbtr.index import watcher
from rbtr.index.embeddings import Embedder
from rbtr.index.gc import run_gc
from rbtr.index.index import Index
from rbtr.index.progress import EmbedOutcome
from rbtr.index.staging import TokenisedChunk

from ..conftest import StubModel, make_commit
from .conftest import serving


class ConcurrencyDetectingStubModel(StubModel):
    """StubModel that counts concurrent calls, and can be held mid-batch.

    Two calls at once is the defect `_gpu_lock` prevents.

    `first_batch_started` is set as the first batch begins, and that
    batch then waits for `first_batch_released`, so a test can act
    while the worker is inside its first batch.
    """

    def __init__(self) -> None:
        super().__init__()
        self._active = 0
        self._lock = threading.Lock()
        self.violations = 0
        self.first_batch_started = threading.Event()
        self.first_batch_released = threading.Event()
        self.first_batch_released.set()

    def embed(
        self,
        text: str | list[str],
        *,
        normalize: bool = True,
        truncate: bool = False,
    ) -> list[float] | list[list[float]]:
        with self._lock:
            if self._active > 0:
                self.violations += 1
            self._active += 1
        try:
            if not self.first_batch_started.is_set():
                self.first_batch_started.set()
                assert self.first_batch_released.wait(20), "inference was not released"
            return super().embed(text, normalize=normalize, truncate=truncate)
        finally:
            with self._lock:
                self._active -= 1


@pytest.fixture
def stub_model() -> ConcurrencyDetectingStubModel:
    return ConcurrencyDetectingStubModel()


@pytest.fixture
def embed_held_at_first_batch(
    stub_model: ConcurrencyDetectingStubModel,
) -> ConcurrencyDetectingStubModel:
    """Hold the worker inside its first batch until a test releases it.

    Requested before `running_daemon`, so the stub is holding by the
    time the worker starts.  Clearing the event later is a race: the
    stub checks it immediately after reporting that the batch began.
    """
    stub_model.first_batch_released.clear()
    return stub_model


@pytest.fixture
def serving_loop() -> Future[asyncio.AbstractEventLoop]:
    return Future()


@pytest.fixture
def batch_finished() -> threading.Event:
    return threading.Event()


@pytest.fixture
def post_batch_work() -> list[threading.Event]:
    return []


@pytest.fixture
def search_started() -> threading.Event:
    return threading.Event()


@pytest.fixture
def gpu_attempted(search_started: threading.Event, mocker: MockerFixture) -> threading.Event:
    attempted = threading.Event()
    acquire = asyncio.Lock.acquire

    async def observed_acquire(lock: asyncio.Lock) -> bool:
        if search_started.is_set():
            attempted.set()
        return await acquire(lock)

    mocker.patch.object(asyncio.Lock, "acquire", observed_acquire)
    return attempted


@pytest.fixture
def observed_native_batches(
    serving_loop: Future[asyncio.AbstractEventLoop],
    batch_finished: threading.Event,
    post_batch_work: list[threading.Event],
    search_started: threading.Event,
    mocker: MockerFixture,
) -> None:
    to_thread = asyncio.to_thread

    async def observe[Result, **Params](
        function: Callable[Params, Result], /, *args: Params.args, **kwargs: Params.kwargs
    ) -> Result:
        if not serving_loop.done():
            serving_loop.set_result(asyncio.get_running_loop())
        finished = threading.Event() if batch_finished.is_set() or search_started.is_set() else None
        if finished is not None:
            post_batch_work.append(finished)
        try:
            result = await to_thread(function, *args, **kwargs)
            if (
                isinstance(function, MethodType)
                and isinstance(function.__self__, Embedder)
                and function.__name__ == "embed"
            ):
                batch_finished.set()
            return result
        finally:
            if finished is not None:
                finished.set()

    mocker.patch.object(asyncio, "to_thread", observe)


@pytest.fixture
def write_lock_held(
    daemon: Daemon,
    embed_held_at_first_batch: ConcurrencyDetectingStubModel,
    fake_repo: str,
    write_started: threading.Event,
    write_released: threading.Event,
    mocker: MockerFixture,
) -> Generator[Daemon]:
    """Hold a real collection after it has acquired foreground writer ownership."""

    def collect(
        index: Index,
        repo_path: str,
        *,
        mode: GcMode,
        refs: list[str],
        dry_run: bool,
        compact: bool = False,
    ) -> GcCounts:
        write_started.set()
        assert write_released.wait(20), "collection was not released"
        return run_gc(index, repo_path, mode=mode, refs=refs, dry_run=dry_run, compact=compact)

    mocker.patch("rbtr.daemon.handlers.run_gc", side_effect=collect)
    with serving(daemon), ThreadPoolExecutor(max_workers=1) as clients:

        def ask() -> Response:
            with DaemonClient(daemon.runtime_dir) as client:
                return client.send(GcRequest(repo_path=fake_repo, mode=GcMode.WATCHED))

        collecting = clients.submit(ask)
        try:
            assert write_started.wait(10), "collection did not start"
            yield daemon
        finally:
            write_released.set()
            assert collecting.result(timeout=10).kind == "gc"


@pytest.fixture
def embedded_snapshot(fake_repo: str, index: Index) -> SnapshotRef:
    """`fake_repo` registered in `index`, at its HEAD commit.

    The id comes back from the registration, so the ref names a repo
    that exists, and the SHA is git's own rather than a literal.
    """
    repo = index.register(fake_repo)
    return repo.at(str(pygit2.Repository(fake_repo).head.target))


@pytest.fixture
def embeddable_index(index: Index, embedded_snapshot: SnapshotRef) -> Index:
    """Store with 50 chunks at `embedded_snapshot` that have no embeddings."""
    chunks: list[TokenisedChunk] = []
    for i in range(50):
        name = f"func_{i}"
        content = f"def func_{i}(x):\n    return x + {i}\n"
        chunks.append(
            TokenisedChunk(
                blob_sha=f"blob_{i}",
                file_path=f"src/mod_{i}.py",
                kind=ChunkKind.FUNCTION,
                name=name,
                content=content,
                content_tokens=tokenise_code(content),
                name_tokens=tokenise_code(name),
                line_start=1,
                line_end=2,
            )
        )
    with index.session(embedded_snapshot.repo) as session:
        for c in chunks:
            session.add_chunk(c)
        session.insert_snapshots(
            [
                FileSnapshot(
                    snapshot_sha=embedded_snapshot.snapshot_sha,
                    file_path=c.file_path,
                    blob_sha=c.blob_sha,
                )
                for c in chunks
            ],
        )
        session.mark_indexed(snapshot_sha=embedded_snapshot.snapshot_sha)
    return index


@pytest.fixture
def daemon(
    runtime_dir: Path,
    embeddable_index: Index,
    stub_model: ConcurrencyDetectingStubModel,
    observed_native_batches: None,
    mocker: MockerFixture,
) -> Daemon:
    """Construct the real runtime before either serving fixture starts work."""
    mocker.patch("rbtr.index.embeddings._load_model", return_value=stub_model)
    return Daemon(
        runtime_dir,
        index=embeddable_index,
        idle_poll_interval=60.0,
        busy_poll_interval=60.0,
    )


@pytest.fixture
def running_daemon(daemon: Daemon) -> Generator[Daemon]:
    with serving(daemon):
        yield daemon


@pytest.fixture
def gpu_daemon(
    daemon: Daemon,
    embed_held_at_first_batch: ConcurrencyDetectingStubModel,
    gpu_attempted: threading.Event,
) -> Generator[Daemon]:
    with serving(daemon):
        yield daemon


@pytest.fixture
def notifications(running_daemon: Daemon) -> Generator[list[Notification]]:
    """Every notification the daemon publishes while the test runs.

    A SUB socket on the daemon's PUB endpoint, drained on a thread: the
    outcome an embed run reports is only visible to a subscriber.
    """
    received: list[Notification] = []
    sub: zmq.Socket[bytes] = zmq.Context.instance().socket(zmq.SUB)
    sub.connect(running_daemon.pub_addr)
    sub.subscribe(b"")
    sub.setsockopt(zmq.RCVTIMEO, 200)

    stop = threading.Event()

    def drain() -> None:
        while not stop.is_set():
            try:
                received.append(notification_adapter.validate_json(sub.recv()))
            except zmq.Again:
                continue

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    yield received
    stop.set()
    reader.join(timeout=2)
    sub.close()


@pytest.fixture
def commit_during_first_batch(
    embed_held_at_first_batch: ConcurrencyDetectingStubModel,
    running_daemon: Daemon,
    fake_repo: str,
) -> str:
    """Move HEAD on while the worker is inside its first batch.

    The stub is holding when the worker reaches its first batch, so the
    commit lands while the job is still running.  If the worker never
    starts, the wait times out and the test fails on its own assertions
    rather than here.

    `make_commit` writes only git objects, so the file goes into the
    working tree too, leaving the tree matching HEAD.
    """
    embed_held_at_first_batch.first_batch_started.wait(timeout=10)
    repo = pygit2.Repository(fake_repo)
    sha = str(
        make_commit(repo, {"later.py": b"x = 1\n"}, parents=[repo.head.peel(pygit2.Commit).id])
    )
    (Path(fake_repo) / "later.py").write_bytes(b"x = 1\n")
    embed_held_at_first_batch.first_batch_released.set()
    return sha


def test_search_works_during_embed(
    gpu_daemon: Daemon,
    fake_repo: str,
    stub_model: ConcurrencyDetectingStubModel,
    search_started: threading.Event,
    gpu_attempted: threading.Event,
    serving_loop: Future[asyncio.AbstractEventLoop],
    post_batch_work: list[threading.Event],
) -> None:
    """A search that needs the held GPU starts inference only after release."""

    def ask() -> SearchResponse:
        with DaemonClient(gpu_daemon.runtime_dir) as client:
            return client.send_or_raise_as(
                SearchResponse, SearchRequest(repo_path=fake_repo, query="func_0", limit=5)
            )

    async def submitted() -> tuple[threading.Event, ...]:
        await asyncio.sleep(0)
        return tuple(post_batch_work)

    with ThreadPoolExecutor(max_workers=1) as clients:
        try:
            assert stub_model.first_batch_started.wait(10), "background inference did not start"
            search_started.set()
            response = clients.submit(ask)
            assert gpu_attempted.wait(10), "search did not attempt GPU acquisition"
            pending = asyncio.run_coroutine_threadsafe(
                submitted(), serving_loop.result(timeout=10)
            ).result(timeout=10)
            for finished in pending:
                assert finished.wait(10), "submitted foreground native work did not finish"
            assert stub_model.violations == 0, "foreground inference overlapped the held batch"
        finally:
            stub_model.first_batch_released.set()
        result = response.result(timeout=20)
    assert "func_0" in {hit.name for hit in result.results}


def test_the_embed_worker_finishes_the_snapshot(
    running_daemon: Daemon,
    embeddable_index: Index,
    embedded_snapshot: SnapshotRef,
) -> None:
    """The worker embeds every chunk in the snapshot.

    Nothing interrupts this one.  Other cases cover standing aside
    and resuming; this one covers a run left alone, which is what a
    paging bug would break.
    """
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if embeddable_index.chunk_counts_for_snapshot(at=embedded_snapshot).is_fully_embedded:
            break
        time.sleep(0.05)

    counts = embeddable_index.chunk_counts_for_snapshot(at=embedded_snapshot)
    assert counts.total == 50, "fixture did not seed the chunks it claims to"
    assert counts.is_fully_embedded, f"{counts.unembedded} of {counts.total} left unembedded"


def test_the_embed_worker_stands_aside_for_a_pending_build(
    running_daemon: Daemon,
    commit_during_first_batch: str,
    embeddable_index: Index,
    embedded_snapshot: SnapshotRef,
    notifications: list[Notification],
    log_output: structlog.testing.LogCapture,
) -> None:
    """The worker returns from its embed job once a build is due, and says so.

    A build wants the job worker, not the GPU, so `_gpu_lock` does
    nothing for it: the worker runs one job at a time, and the build
    starts only when the embed job returns.  The commit lands while the
    worker is inside its first batch, so the poll after that batch is
    the first chance the loop has to notice.

    What it publishes matters as much as when it stops: every ending
    sends the same notification, and a subscriber can only tell a
    finished run from a yielded one by the outcome it carries.
    """
    counts = embeddable_index.chunk_counts_for_snapshot(at=embedded_snapshot)
    assert watcher.poll_worktree(embeddable_index) == [], (
        "a dirty worktree would stand the loop aside for the wrong reason"
    )
    deadline = time.monotonic() + 10.0
    ended: list[EmbedEndedNotification] = []
    while time.monotonic() < deadline:
        ended = [n for n in notifications if isinstance(n, EmbedEndedNotification)]
        if ended:
            break
        time.sleep(0.01)

    assert ended, "the embed run published no ending"
    assert ended[0].outcome == EmbedOutcome.STOOD_ASIDE
    assert ended[0].embedded < ended[0].chunks
    stood_aside = [e for e in log_output.entries if e["event"] == "embedding_preempted"]
    assert stood_aside, "the embed job ran to completion with a build waiting"
    assert stood_aside[0]["done"] == config.embedding_batch_size
    assert stood_aside[0]["total"] == counts.total


def test_the_embed_worker_resumes_where_it_stood_aside(
    running_daemon: Daemon,
    commit_during_first_batch: str,
    embeddable_index: Index,
    embedded_snapshot: SnapshotRef,
    log_output: structlog.testing.LogCapture,
) -> None:
    """The worker finishes the snapshot, embedding the rest exactly once.

    This is what lets the worker drop a job rather than pause it: the
    remainder comes from `chunks.embedding`, so the build runs in
    between and the next job embeds only what the first run missed.

    Waits on what the runs report rather than on the counts: a run
    writes its chunks before it logs, so the store reads complete first.
    """
    counts = embeddable_index.chunk_counts_for_snapshot(at=embedded_snapshot)
    # A run reports through `embedding_preempted` when it stands aside
    # and `embedded_chunks` when it reaches the end; both carry the ref.
    runs: list[int] = []
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        runs = [
            e["done"]
            for e in log_output.entries
            if e["event"] in {"embedded_chunks", "embedding_preempted"}
            and e.get("ref") == embedded_snapshot.snapshot_sha
        ]
        if sum(runs) >= counts.total:
            break
        time.sleep(0.05)

    final = embeddable_index.chunk_counts_for_snapshot(at=embedded_snapshot)
    assert final.is_fully_embedded, f"{final.unembedded} of {final.total} left unembedded"
    # The runs for this snapshot must add up to its chunks and no more:
    # more would mean a resumed run re-embedded what was already there.
    assert len(runs) > 1, f"expected the job to take more than one run, got {runs}"
    assert sum(runs) == final.total, f"runs embedded {sum(runs)} of {final.total}: {runs}"


def test_no_writer_commits_while_the_write_lock_is_held(
    write_lock_held: Daemon,
    embed_held_at_first_batch: ConcurrencyDetectingStubModel,
    write_released: threading.Event,
    batch_finished: threading.Event,
    serving_loop: Future[asyncio.AbstractEventLoop],
    post_batch_work: list[threading.Event],
    embeddable_index: Index,
    embedded_snapshot: SnapshotRef,
    fake_repo: str,
    rpc_peer: zmq.Socket[bytes],
    mocker: MockerFixture,
) -> None:
    """A foreground collection excludes watch writes and embedding commits."""
    repo = embeddable_index.repo(fake_repo)
    sessions = mocker.spy(embeddable_index, "session")
    try:
        rpc_peer.send_multipart(
            [
                b"watch",
                b"",
                WatchRequest(repo_path=fake_repo, refs=["main"]).model_dump_json().encode(),
            ]
        )
        rpc_peer.send_multipart([b"fence", b"", DaemonConfigRequest().model_dump_json().encode()])
        frames = rpc_peer.recv_multipart()
        assert frames[:2] == [b"fence", b""]
        assert response_adapter.validate_json(frames[-1]).kind == "daemon_config"

        embed_held_at_first_batch.first_batch_released.set()
        assert batch_finished.wait(10), "native embedding did not finish"

        async def advance() -> tuple[threading.Event, ...]:
            await asyncio.to_thread(lambda: None)
            return tuple(post_batch_work)

        pending = asyncio.run_coroutine_threadsafe(
            advance(), serving_loop.result(timeout=10)
        ).result(timeout=10)
        for finished in pending:
            assert finished.wait(10), "post-batch native work did not finish"
        counts = embeddable_index.chunk_counts_for_snapshot(at=embedded_snapshot)
        assert counts.unembedded == counts.total
        assert "main" not in embeddable_index.watched_refs(repo)
        assert not sessions.called, "a write session was requested during collection"
    finally:
        write_released.set()
        embed_held_at_first_batch.first_batch_released.set()

    frames = rpc_peer.recv_multipart()
    assert frames[:2] == [b"watch", b""]
    assert response_adapter.validate_json(frames[-1]).kind == "watch_set"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if embeddable_index.chunk_counts_for_snapshot(at=embedded_snapshot).is_fully_embedded:
            break
        time.sleep(0.05)

    assert "main" in embeddable_index.watched_refs(repo)
    final = embeddable_index.chunk_counts_for_snapshot(at=embedded_snapshot)
    assert final.is_fully_embedded, f"{final.unembedded} of {final.total} left unembedded"
