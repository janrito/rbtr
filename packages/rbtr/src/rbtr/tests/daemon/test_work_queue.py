"""Work selection and startup recovery against persisted indexing data."""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from pathlib import Path

import pygit2
import pytest
import zmq
from pytest_cases import fixture, parametrize_with_cases

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    IndexErrorNotification,
    WatchRequest,
    notification_adapter,
)
from rbtr.daemon.server import Daemon
from rbtr.errors import BuildStopped
from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index
from rbtr.index.indexer import BuildJob, EmbeddingEnded, EmbedJob, Indexer, IndexerOutput

from ..conftest import make_commit
from .cases_work_queue import NextJobScenario
from .conftest import serving, subscribed


def _stop_at_edges(stop: threading.Event, phase: str, _done: int, _total: int) -> None:
    """Stop a build once its chunks are written, before it marks the snapshot indexed."""
    if phase == "edges":
        stop.set()


@fixture
@parametrize_with_cases("scenario", cases=".cases_work_queue", has_tag="next_job")
def next_job(scenario: NextJobScenario) -> NextJobScenario:
    return scenario


@pytest.fixture
def next_job_shas(
    next_job: NextJobScenario,
    index: Index,
    fake_repo: str,
    second_repo: str,
    stubbed_embedder: Embedder,
) -> dict[str, str]:
    """Each of the scenario's snapshots built, and embedded where it says, in list order.

    Returns each snapshot's label mapped to its commit.
    """
    repos = {"a": fake_repo, "b": second_repo}
    shas: dict[str, str] = {}
    for n, snap in enumerate(next_job.snapshots):
        path = repos[snap.repo]
        source = f"""\
def fn{n}():
    pass
""".encode()
        files = {f"f{n}.py": source}
        sha = str(make_commit(pygit2.Repository(path), files, ref=f"refs/heads/{snap.label}"))
        if snap.indexed:
            build_index(path, sha, index)
        else:
            stop = threading.Event()
            with pytest.raises(BuildStopped):
                build_index(
                    path,
                    sha,
                    index,
                    stop=stop,
                    on_progress=functools.partial(_stop_at_edges, stop),
                )
        if snap.embedded:
            embed_index(index, at=index.register(path).at(sha), embedder=stubbed_embedder)
        shas[snap.label] = sha
    return shas


@pytest.fixture
def index_for_indexer(next_job_shas: dict[str, str], index: Index) -> Index:
    return index


def test_startup_recovery_runs_the_selected_embed_work(
    next_job: NextJobScenario,
    next_job_shas: dict[str, str],
    index: Index,
    indexer: Indexer,
    indexer_outputs: list[IndexerOutput],
    indexer_loop: asyncio.AbstractEventLoop,
) -> None:
    """Recovery selects the newest eligible snapshot and embeds it without a wake."""
    expected = None if next_job.expected_ref is None else next_job_shas[next_job.expected_ref]
    job = indexer.next_job()
    assert (job.at.snapshot_sha if isinstance(job, EmbedJob) else job) == expected

    async def run() -> None:
        if expected is None:
            indexer.request_shutdown()
        async with asyncio.timeout(20):
            await indexer.run()

    asyncio.run_coroutine_threadsafe(run(), indexer_loop).result()
    ended = [event for event in indexer_outputs if isinstance(event, EmbeddingEnded)]
    assert [event.at.snapshot_sha for event in ended] == ([] if expected is None else [expected])
    if isinstance(job, EmbedJob):
        assert index.chunk_counts_for_snapshot(at=job.at).is_fully_embedded


def test_startup_backfills_a_head_watch_for_every_repo(
    index: Index,
    runtime_dir: Path,
) -> None:
    """Once serving, the daemon gives each registered repo a HEAD watch."""
    repo = index.register("/test/repo")
    server = Daemon(runtime_dir, index=index, idle_poll_interval=60.0, busy_poll_interval=60.0)

    with serving(server):
        deadline = time.monotonic() + 10
        while index.watched_refs(repo) != ["HEAD"]:
            assert time.monotonic() < deadline, "no HEAD watch was seeded"
            time.sleep(0.05)


@pytest.fixture
def dirty_unindexed_repo(tmp_path: Path) -> str:
    """Repo with a committed file, then dirtied — HEAD and tree both unindexed."""
    repo_dir = tmp_path / "dirty"
    repo = pygit2.init_repository(str(repo_dir), bare=False, initial_head="main")
    sig = pygit2.Signature("t", "t@t.t")
    (repo_dir / "f.py").write_text("x = 1\n")
    repo.index.add("f.py")
    repo.index.write()
    repo.create_commit("refs/heads/main", sig, sig, "init", repo.index.write_tree(), [])
    (repo_dir / "f.py").write_text("x = 2\n")
    return str(repo_dir)


@pytest.fixture
def dirty_index(dirty_unindexed_repo: str, index: Index) -> Index:
    """Store with the dirty repo registered and watching HEAD, as `run` seeds it; nothing indexed."""
    repo = index.register(dirty_unindexed_repo)
    with index.session(repo) as ws:
        ws.add_watched_refs(["HEAD"])
    return index


def test_find_next_job_prefers_stale_watched_ref(
    dirty_index: Index,
    dirty_unindexed_repo: str,
    indexer_loop: asyncio.AbstractEventLoop,
) -> None:
    """`_find_next_job` derives a build from the stale watched ref.

    The repo's HEAD (watched) is un-indexed *and* its worktree
    is dirty, so both are stale; returning HEAD's SHA proves the build
    is derived from `watched_refs` and that watched refs win over the
    dirty worktree (the core regression — impossible before).
    """

    async def output(_event: IndexerOutput) -> None:
        pass

    async def build() -> Indexer:
        return Indexer(
            dirty_index,
            embedder=None,
            gpu_lock=asyncio.Lock(),
            write_sem=asyncio.Semaphore(1),
            stop=threading.Event(),
            on_output=output,
        )

    service = asyncio.run_coroutine_threadsafe(build(), indexer_loop).result(timeout=10)
    head = str(pygit2.Repository(dirty_unindexed_repo).head.target)
    job = service.next_job()
    assert isinstance(job, BuildJob)
    assert job.refs == (head,)


def test_a_build_that_fails_is_reported_once_and_not_retried(
    runtime_dir: Path, seeded_index: Index, fake_repo: str
) -> None:
    """A watched ref whose build raises (here, a commit whose tree is gone)
    gets one `index_error`, and is not built again on each poll."""
    repo = pygit2.Repository(fake_repo)
    broken = make_commit(
        repo,
        {"broken.py": b"x = 1\n"},
        parents=[repo.head.peel(pygit2.Commit).id],
        ref="refs/heads/broken",
    )
    tree = str(repo[broken].peel(pygit2.Commit).tree_id)
    (Path(repo.path) / "objects" / tree[:2] / tree[2:]).unlink()

    server = Daemon(
        runtime_dir, index=seeded_index, idle_poll_interval=0.05, busy_poll_interval=0.05
    )
    errors: list[IndexErrorNotification] = []

    def receive_for(sub: zmq.Socket[bytes], seconds: float, *, until_error: bool) -> None:
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            if not sub.poll(int(remaining * 1000)):
                return
            notification = notification_adapter.validate_json(sub.recv())
            if isinstance(notification, IndexErrorNotification):
                errors.append(notification)
                if until_error:
                    return

    with serving(server), subscribed(server, fake_repo) as sub:
        with DaemonClient(server.runtime_dir) as client:
            client.send(WatchRequest(repo_path=fake_repo, refs=["broken"]))
        receive_for(sub, 30.0, until_error=True)
        receive_for(sub, 1.0, until_error=False)  # twenty polls

    assert len(errors) == 1
