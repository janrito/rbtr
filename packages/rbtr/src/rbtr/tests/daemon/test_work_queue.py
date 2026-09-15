"""What the daemon decides to work on, and when it wakes to do it.

Startup recovery sets `DaemonServer._wake` when an index has embedding
left to do; `_find_next_job` then picks which build or embed runs next.
Both read the same scenarios, because both are asking the index the
same question.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.daemon.messages import BuildJob, EmbedJob
from rbtr.daemon.server import DaemonServer
from rbtr.domain.models import FileSnapshot, SnapshotRef
from rbtr.index.store import IndexStore

from ..index.conftest import make_chunk
from .cases_work_queue import NextJobScenario

# ── Fixtures ─────────────────────────────────────────────────────────


@fixture
@parametrize_with_cases("scenario", cases=".cases_work_queue", has_tag="next_job")
def next_job_store(
    scenario: NextJobScenario, store: IndexStore
) -> tuple[IndexStore, NextJobScenario]:
    for n, snap in enumerate(scenario.snapshots):
        with store.session() as ws:
            repo_id = ws.register_repo(snap.repo_path)
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
                repo_id=repo_id,
            )
            if snap.embedded:
                ws.update_embeddings([chunk.id], [[0.1, 0.2, 0.3]])
            if snap.indexed:
                ws.mark_indexed(at=SnapshotRef(repo_id=repo_id, snapshot_sha=snap.snapshot_sha))
    return store, scenario


# ── Startup recovery ────────────────────────────────────────────────


def test_startup_recovery_wakes_the_worker_for_outstanding_embeds(
    next_job_store: tuple[IndexStore, NextJobScenario],
    runtime_dir: Path,
) -> None:
    """A daemon starting on an index with embed work left wakes its worker.

    Recovery and job selection ask the same question of the index, so
    they read the same scenarios: the wake event follows whether any
    snapshot has chunks left to embed.
    """
    store, scenario = next_job_store

    server = DaemonServer(
        runtime_dir, store=store, idle_poll_interval=60.0, busy_poll_interval=60.0
    )

    assert server._wake.is_set() == (scenario.expected_ref is not None)


def test_startup_backfills_a_head_watch_for_every_repo(
    store: IndexStore,
    runtime_dir: Path,
) -> None:
    """`_backfill_head_watches` gives each registered repo a HEAD watch."""
    with store.session() as ws:
        ws.register_repo("/test/repo")

    DaemonServer(runtime_dir, store=store, idle_poll_interval=60.0, busy_poll_interval=60.0)

    assert store.list_watched_refs(store.resolve_repo("/test/repo")) == ["HEAD"]


# ── Watched-ref builds & HEAD backfill ───────────────────────────────


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
def dirty_store(dirty_unindexed_repo: str, store: IndexStore) -> IndexStore:
    """Store with the dirty repo registered, nothing indexed."""
    with store.session() as ws:
        ws.register_repo(dirty_unindexed_repo)
    return store


def test_find_next_job_prefers_stale_watched_ref(
    dirty_store: IndexStore,
    dirty_unindexed_repo: str,
    runtime_dir: Path,
) -> None:
    """`_find_next_job` derives a build from the stale watched ref.

    The repo's HEAD (backfilled watch) is un-indexed *and* its worktree
    is dirty, so both are stale; returning HEAD's SHA proves the build
    is derived from `watched_refs` and that watched refs win over the
    dirty worktree (the core regression — impossible before).
    """
    server = DaemonServer(
        runtime_dir, store=dirty_store, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    head = str(pygit2.Repository(dirty_unindexed_repo).head.target)
    job = server._find_next_job()
    assert isinstance(job, BuildJob)
    assert job.refs == (head,)


# ── Which job comes next ─────────────────────────────────────────────


def test_find_next_job_picks_the_embed_work(
    next_job_store: tuple[IndexStore, NextJobScenario],
    runtime_dir: Path,
) -> None:
    """`_find_next_job` returns the embed job that is due.

    These repo paths are absent from disk, so ref resolution skips the
    watched-ref and worktree branches and the embed branch decides.
    """
    store, scenario = next_job_store
    server = DaemonServer(
        runtime_dir, store=store, idle_poll_interval=60.0, busy_poll_interval=60.0
    )

    job = server._find_next_job()

    assert (job.ref if isinstance(job, EmbedJob) else job) == scenario.expected_ref
