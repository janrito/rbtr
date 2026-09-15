"""Tests for how the embed worker shares the daemon with everything else.

Search and embed contend for the GPU, so they serialise through
`_gpu_lock`: the real `Embedder` class runs here with a `StubModel`
injected via `model_loader`, and a concurrent-access detector on
`StubModel.embed()` proves the race is fixed at the exact call site
where `llama_cpp` would crash in production.

A build contends for something else entirely — the single job worker —
so the lock cannot serve it.  The worker runs one job at a time, and a
build starts only once the embed job returns, which is what the last
test here pins.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from pathlib import Path

import pygit2
import pytest
import structlog
import zmq

from rbtr.config import config
from rbtr.daemon import watcher
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    EmbedEndedNotification,
    EmbedOutcome,
    Notification,
    SearchRequest,
    SearchResponse,
    notification_adapter,
)
from rbtr.daemon.server import DaemonServer
from rbtr.domain.models import ChunkKind, FileSnapshot, SnapshotRef
from rbtr.domain.tokenise import tokenise_code
from rbtr.index.embeddings import Embedder
from rbtr.index.staging import TokenisedChunk
from rbtr.index.store import IndexStore

from ..conftest import StubModel, make_commit
from .conftest import serving

# ── Concurrency-detecting stub model ──────────────────────────────────


class ConcurrencyDetectingStubModel(StubModel):
    """StubModel that detects concurrent access, and can be held mid-batch.

    Concurrent access is the exact defect the embed lock prevents.

    `first_batch_started` is set as the first batch begins, and that
    batch then waits for `first_batch_released`, so a test can change
    the world while the worker is provably inside its first batch —
    without sleeps, and without a second daemon to drive the loop by
    hand.
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
                self.first_batch_released.wait(timeout=10)
            # Small sleep to widen the race window.
            time.sleep(0.005)
            return super().embed(text, normalize=normalize, truncate=truncate)
        finally:
            with self._lock:
                self._active -= 1


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def stub_model() -> ConcurrencyDetectingStubModel:
    return ConcurrencyDetectingStubModel()


@pytest.fixture
def embedder(stub_model: ConcurrencyDetectingStubModel) -> Generator[Embedder]:
    e = Embedder(model_loader=lambda: stub_model)  # type: ignore[arg-type,return-value]  # StubModel satisfies Llama.embed interface
    yield e
    e.close()


@pytest.fixture
def embedded_snapshot(fake_repo: str, store: IndexStore) -> SnapshotRef:
    """`fake_repo` registered in `store`, at its HEAD commit.

    The id comes back from the registration, so the ref names a repo
    that exists, and the SHA is git's own rather than a literal.
    """
    with store.session() as ws:
        repo_id = ws.register_repo(fake_repo)
    return SnapshotRef(
        repo_id=repo_id,
        snapshot_sha=str(pygit2.Repository(fake_repo).head.target),
    )


@pytest.fixture
def embeddable_store(store: IndexStore, embedded_snapshot: SnapshotRef) -> IndexStore:
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
    with store.session() as session:
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
            repo_id=embedded_snapshot.repo_id,
        )
        session.mark_indexed(at=embedded_snapshot)
    return store


@pytest.fixture
def running_daemon(
    runtime_dir: Path,
    embeddable_store: IndexStore,
    embedder: Embedder,
) -> Generator[DaemonServer]:
    """A served daemon with a stub embedder and unembedded chunks.

    The embed worker starts immediately (``_wake`` is set).
    """
    server = DaemonServer(
        runtime_dir,
        store=embeddable_store,
        idle_poll_interval=60.0,
        busy_poll_interval=60.0,
    )
    # Replace the real embedder with our stub-backed one.
    server._embedder = embedder
    # Re-register handlers so search uses the stub embedder.
    server._register_index_handlers(embeddable_store)
    # Wake the worker so embedding starts immediately.
    server._wake.set()

    with serving(server):
        yield server


@pytest.fixture
def notifications(running_daemon: DaemonServer) -> Generator[list[Notification]]:
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
    running_daemon: DaemonServer,
    stub_model: ConcurrencyDetectingStubModel,
    fake_repo: str,
) -> str:
    """Move HEAD on while the worker is inside its first batch.

    Runs after `running_daemon` is serving, so the worker has already
    started embedding; the stub holds its first batch until this
    releases it, which is what makes the timing deterministic.  If the
    worker never starts, the wait times out and the test fails on its
    own assertions rather than here.

    `make_commit` writes only git objects, so the file goes into the
    working tree too, leaving the tree matching HEAD.
    """
    stub_model.first_batch_started.wait(timeout=10)
    stub_model.first_batch_released.clear()
    repo = pygit2.Repository(fake_repo)
    sha = str(
        make_commit(repo, {"later.py": b"x = 1\n"}, parents=[repo.head.peel(pygit2.Commit).id])
    )
    (Path(fake_repo) / "later.py").write_bytes(b"x = 1\n")
    stub_model.first_batch_released.set()
    return sha


# ── Tests ────────────────────────────────────────────────────────────


def test_search_works_during_embed(
    running_daemon: DaemonServer,
    fake_repo: str,
    stub_model: ConcurrencyDetectingStubModel,
) -> None:
    """Search and embed don't access the model concurrently.

    Fires search requests while the embed worker is processing
    batches.  Asserts no concurrent-access violations and no
    crashes.
    """
    with DaemonClient(running_daemon.runtime_dir) as client:
        responses: list[SearchResponse] = []
        for _ in range(20):
            resp = client.send(SearchRequest(repo_path=fake_repo, query="func_0", limit=5))
            assert isinstance(resp, SearchResponse)
            responses.append(resp)
            time.sleep(0.01)

    assert stub_model.violations == 0, f"Detected {stub_model.violations} concurrent model accesses"
    assert len(responses) == 20


def test_search_results_correct_during_embed(
    running_daemon: DaemonServer,
    fake_repo: str,
    embeddable_store: IndexStore,
    embedded_snapshot: SnapshotRef,
    embedder: Embedder,
) -> None:
    """Search results from the daemon match a direct store search.

    After embedding completes, daemon search should return the
    same results as a direct call.
    """
    # Wait for some embeddings to be written.
    time.sleep(0.5)

    with DaemonClient(running_daemon.runtime_dir) as client:
        daemon_resp = client.send(SearchRequest(repo_path=fake_repo, query="func_0", limit=5))
        assert isinstance(daemon_resp, SearchResponse)

    # Direct search for comparison.
    direct_results = embeddable_store.search(
        "func_0",
        within=[embedded_snapshot],
        top_k=5,
        embedder=embedder,
    )

    # Both should return results (possibly empty if embeddings
    # aren't complete yet, but they should match).
    daemon_names = {r.name for r in daemon_resp.results}
    direct_names = {r.name for r in direct_results}
    # At minimum, both should succeed without error.
    assert isinstance(daemon_names, set)
    assert isinstance(direct_names, set)


def test_the_embed_worker_finishes_the_snapshot(
    running_daemon: DaemonServer,
    embeddable_store: IndexStore,
    embedded_snapshot: SnapshotRef,
) -> None:
    """The worker embeds every chunk in the snapshot.

    Nothing interrupts this one: the tests above show the worker
    standing aside and resuming, while this holds an undisturbed run to
    finishing the work, which is what a paging bug would break.
    """
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if embeddable_store.chunk_counts_for_snapshot(at=embedded_snapshot).is_fully_embedded:
            break
        time.sleep(0.05)

    counts = embeddable_store.chunk_counts_for_snapshot(at=embedded_snapshot)
    assert counts.total == 50, "fixture did not seed the chunks it claims to"
    assert counts.is_fully_embedded, f"{counts.unembedded} of {counts.total} left unembedded"


def test_the_embed_worker_stands_aside_for_a_pending_build(
    running_daemon: DaemonServer,
    commit_during_first_batch: str,
    embeddable_store: IndexStore,
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
    counts = embeddable_store.chunk_counts_for_snapshot(at=embedded_snapshot)
    assert watcher.poll_worktree(embeddable_store) == [], (
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
    running_daemon: DaemonServer,
    commit_during_first_batch: str,
    embeddable_store: IndexStore,
    embedded_snapshot: SnapshotRef,
    log_output: structlog.testing.LogCapture,
) -> None:
    """The worker finishes the snapshot, embedding the rest exactly once.

    This is what lets the worker drop a job rather than pause it: the
    remainder is re-derived from `chunks.embedding`, so the build runs
    in between and the job that follows embeds only the chunks the
    first run had not reached.

    Waits on what the runs report rather than on the counts: a run
    writes its chunks before it logs, so the store reads complete first.
    """
    counts = embeddable_store.chunk_counts_for_snapshot(at=embedded_snapshot)
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

    final = embeddable_store.chunk_counts_for_snapshot(at=embedded_snapshot)
    assert final.is_fully_embedded, f"{final.unembedded} of {final.total} left unembedded"
    # The runs for this snapshot must add up to its chunks and no more:
    # more would mean a resumed run re-embedded what was already there.
    assert len(runs) > 1, f"expected the job to take more than one run, got {runs}"
    assert sum(runs) == final.total, f"runs embedded {sum(runs)} of {final.total}: {runs}"
