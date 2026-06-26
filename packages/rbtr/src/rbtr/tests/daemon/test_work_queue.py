"""Tests for startup recovery.

Recovery sets `DaemonServer._wake` so the DB-polling worker
picks up un-embedded commits on startup.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.messages import BuildJob
from rbtr.daemon.server import DaemonServer
from rbtr.index.models import Snapshot
from rbtr.index.store import IndexStore

from ..index.conftest import make_chunk

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def recovery_store_unembedded() -> Generator[IndexStore]:
    """Store with one indexed but unembedded commit."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo("/test/repo")
        ws.add_chunk(make_chunk("chunk1", name="foo", path="test.py", blob="blob1"))
        ws.insert_snapshots(
            [Snapshot(commit_sha="sha1", file_path="test.py", blob_sha="blob1")],
            repo_id=repo_id,
        )
        ws.mark_indexed(repo_id, "sha1")
    yield store
    store.close()


@pytest.fixture
def recovery_store_fully_embedded() -> Generator[IndexStore]:
    """Store with one indexed and fully embedded commit."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo("/test/repo")
        ws.add_chunk(make_chunk("chunk1", name="foo", path="test.py", blob="blob1"))
        ws.insert_snapshots(
            [Snapshot(commit_sha="sha1", file_path="test.py", blob_sha="blob1")],
            repo_id=repo_id,
        )
        ws.update_embeddings(["chunk1"], [[0.1, 0.2, 0.3]])
        ws.mark_indexed(repo_id, "sha1")
    yield store
    store.close()


# ── Startup recovery ────────────────────────────────────────────────


def test_startup_recovery_enqueues_embed_jobs(
    recovery_store_unembedded: IndexStore,
    runtime_dir: Path,
) -> None:
    """Unembedded commit → wake event set; construction backfills a HEAD watch."""
    server = DaemonServer(
        runtime_dir,
        store=recovery_store_unembedded,
        idle_poll_interval=60.0,
        busy_poll_interval=60.0,
    )
    assert server._wake.is_set()
    # `_backfill_head_watches` seeds a HEAD watch for every registered repo.
    repo_id = recovery_store_unembedded.resolve_repo("/test/repo")
    assert recovery_store_unembedded.list_watched_refs(repo_id) == ["HEAD"]


def test_startup_recovery_skips_fully_embedded(
    recovery_store_fully_embedded: IndexStore,
    runtime_dir: Path,
) -> None:
    """Fully embedded commit → wake event not set."""
    server = DaemonServer(
        runtime_dir,
        store=recovery_store_fully_embedded,
        idle_poll_interval=60.0,
        busy_poll_interval=60.0,
    )
    assert not server._wake.is_set()


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
def dirty_store(dirty_unindexed_repo: str) -> Generator[IndexStore]:
    """Store with the dirty repo registered, nothing indexed."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        ws.register_repo(dirty_unindexed_repo)
    yield store
    store.close()


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
