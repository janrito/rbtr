"""Work selection and startup recovery against persisted indexing data."""

from __future__ import annotations

import asyncio
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
from rbtr.domain.models import FileSnapshot
from rbtr.index.index import Index
from rbtr.index.indexer import BuildJob, EmbeddingEnded, EmbedJob, Indexer, IndexerOutput

from ..conftest import make_commit
from ..index.conftest import make_chunk
from .cases_work_queue import NextJobScenario
from .conftest import serving


@fixture
@parametrize_with_cases("scenario", cases=".cases_work_queue", has_tag="next_job")
def next_job_index(scenario: NextJobScenario, index: Index) -> tuple[Index, NextJobScenario]:
    for n, snap in enumerate(scenario.snapshots):
        repo = index.register(snap.repo_path)
        with index.session(repo) as ws:
            chunk = make_chunk(f"c{n}", name=f"fn{n}", path=f"f{n}.py", blob=f"blob{n}")
            ws.add_chunk(chunk)
            ws.insert_snapshots(
                [
                    FileSnapshot(
                        snapshot_sha=snap.snapshot_sha,
                        file_path=f"f{n}.py",
                        blob_sha=f"blob{n}",
                    )
                ],
            )
            if snap.embedded:
                ws.update_embeddings([chunk.id], [[0.1, 0.2, 0.3]])
            if snap.indexed:
                ws.mark_indexed(snapshot_sha=snap.snapshot_sha)
    return index, scenario


@pytest.fixture
def index_for_indexer(next_job_index: tuple[Index, NextJobScenario]) -> Index:
    return next_job_index[0]


def test_startup_recovery_runs_the_selected_embed_work(
    next_job_index: tuple[Index, NextJobScenario],
    indexer: Indexer,
    indexer_outputs: list[IndexerOutput],
) -> None:
    """Recovery selects the newest eligible snapshot and embeds it without a wake."""
    index, scenario = next_job_index
    job = indexer.next_job()
    assert (job.at.snapshot_sha if isinstance(job, EmbedJob) else job) == scenario.expected_ref

    async def run() -> None:
        if scenario.expected_ref is None:
            indexer.request_shutdown()
        async with asyncio.timeout(20):
            await indexer.run()

    asyncio.run(run())
    ended = [event for event in indexer_outputs if isinstance(event, EmbeddingEnded)]
    assert [event.at.snapshot_sha for event in ended] == (
        [] if scenario.expected_ref is None else [scenario.expected_ref]
    )
    if isinstance(job, EmbedJob):
        assert index.chunk_counts_for_snapshot(at=job.at).is_fully_embedded


def test_startup_backfills_a_head_watch_for_every_repo(
    index: Index,
    runtime_dir: Path,
) -> None:
    """`_backfill_head_watches` gives each registered repo a HEAD watch."""
    index.register("/test/repo")

    Daemon(runtime_dir, index=index, idle_poll_interval=60.0, busy_poll_interval=60.0)

    assert index.watched_refs(index.repo("/test/repo")) == ["HEAD"]


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
    """Store with the dirty repo registered, nothing indexed."""
    index.register(dirty_unindexed_repo)
    return index


def test_find_next_job_prefers_stale_watched_ref(
    dirty_index: Index,
    dirty_unindexed_repo: str,
) -> None:
    """`_find_next_job` derives a build from the stale watched ref.

    The repo's HEAD (backfilled watch) is un-indexed *and* its worktree
    is dirty, so both are stale; returning HEAD's SHA proves the build
    is derived from `watched_refs` and that watched refs win over the
    dirty worktree (the core regression — impossible before).
    """

    async def output(_event: IndexerOutput) -> None:
        pass

    service = Indexer(
        dirty_index,
        embedder=None,
        gpu_lock=asyncio.Lock(),
        write_sem=asyncio.Semaphore(1),
        stop=threading.Event(),
        on_output=output,
    )
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
    sub: zmq.Socket[bytes] = zmq.Context.instance().socket(zmq.SUB)
    errors: list[IndexErrorNotification] = []

    def receive_for(seconds: float, *, until_error: bool) -> None:
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            if not sub.poll(int(remaining * 1000)):
                return
            notification = notification_adapter.validate_json(sub.recv())
            if isinstance(notification, IndexErrorNotification):
                errors.append(notification)
                if until_error:
                    return

    with serving(server):
        sub.connect(server.pub_addr)
        sub.subscribe(b"")
        time.sleep(0.5)  # the subscription reaches the daemon
        with DaemonClient(server.runtime_dir) as client:
            client.send(WatchRequest(repo_path=fake_repo, refs=["broken"]))
        receive_for(10.0, until_error=True)
        receive_for(1.0, until_error=False)  # twenty polls
    sub.close()

    assert len(errors) == 1
