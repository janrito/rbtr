"""Tests for `rbtr.index.gc.run_gc`.

Covers every mode against a tiny seeded repo with three commits.
Uses real pygit2 repositories and a real in-memory IndexStore so
we exercise the actual code path, not mocks.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.messages import GcMode
from rbtr.errors import RbtrError
from rbtr.git import worktree_tree_sha
from rbtr.index.gc import run_gc
from rbtr.index.models import ChunkKind, Snapshot, TokenisedChunk
from rbtr.index.store import IndexStore


@dataclass(frozen=True)
class GcFixture:
    """A real git repo, an in-memory store, and the three SHAs indexed."""

    repo: pygit2.Repository
    store: IndexStore
    repo_id: int
    c1: str  # first commit on main, tagged as v1
    c2: str  # middle commit, no longer reachable by any named ref
    c3: str  # HEAD on main


@pytest.fixture
def gc_signature() -> pygit2.Signature:
    return pygit2.Signature("t", "t@t.t")


@pytest.fixture
def gc_repo(
    tmp_path: Path, gc_signature: pygit2.Signature
) -> tuple[pygit2.Repository, str, str, str]:
    """Three-commit repo.  Tags `v1` at the first commit so KEEP_REFS has
    a non-HEAD ref to preserve; leaves the middle commit unreachable so
    HEAD_ONLY has something to drop."""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=False, initial_head="main")
    shas: list[str] = []
    for i in range(3):
        tb = repo.TreeBuilder()
        tb.insert(
            "a.py",
            repo.create_blob(f"x = {i + 1}\n".encode()),
            pygit2.GIT_FILEMODE_BLOB,
        )
        parents = [repo.head.target] if not repo.head_is_unborn else []
        new = repo.create_commit(
            "refs/heads/main",
            gc_signature,
            gc_signature,
            f"c{i}",
            tb.write(),
            parents,
        )
        shas.append(str(new))
    repo.create_reference("refs/tags/v1", shas[0])
    return repo, shas[0], shas[1], shas[2]


@pytest.fixture
def gc(
    gc_repo: tuple[pygit2.Repository, str, str, str],
) -> Generator[GcFixture]:
    """Index all three commits into a fresh in-memory store."""
    repo, c1, c2, c3 = gc_repo
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo("/repo")
    for i, sha in enumerate((c1, c2, c3)):
        chunk = TokenisedChunk(
            id=f"c{i}",
            repo_id=repo_id,
            blob_sha=f"b{i}",
            file_path="a.py",
            kind=ChunkKind.FUNCTION,
            name="f",
            content="",
            line_start=1,
            line_end=1,
        )
        with store.session() as ws:
            ws.add_chunk(chunk)
            ws.insert_snapshots(
                [Snapshot(commit_sha=sha, file_path="a.py", blob_sha=f"b{i}")], repo_id=repo_id
            )
            ws.mark_indexed(repo_id, sha)
    yield GcFixture(repo=repo, store=store, repo_id=repo_id, c1=c1, c2=c2, c3=c3)
    store.close()


# ── HEAD_ONLY ────────────────────────────────────────────────────────


def test_head_only_keeps_head_and_drops_rest(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False
    )
    assert counts.commits == 2  # c1 and c2 dropped
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is False
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


# ── KEEP_REFS ────────────────────────────────────────────────────────


def test_keep_refs_preserves_tag_and_head(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.KEEP_REFS, refs=[], dry_run=False
    )
    assert counts.commits == 1  # only c2 (unreachable) dropped
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True  # tag v1
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True  # HEAD
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


# ── KEEP ─────────────────────────────────────────────────────────────


def test_keep_preserves_listed_refs_and_head(gc: GcFixture) -> None:
    # Keep v1 (c1). HEAD (c3) is kept implicitly.
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.KEEP, refs=["v1"], dry_run=False
    )
    assert counts.commits == 1
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


def test_keep_with_no_refs_is_head_only(gc: GcFixture) -> None:
    run_gc(gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.KEEP, refs=[], dry_run=False)
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True  # HEAD kept
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is False
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


# ── DROP ─────────────────────────────────────────────────────────────


def test_drop_removes_only_listed_refs(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.DROP, refs=["v1"], dry_run=False
    )
    assert counts.commits == 1
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is False
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is True


def test_drop_head_is_allowed(gc: GcFixture) -> None:
    """DROP does not implicitly keep HEAD — if the user says drop HEAD, do it."""
    run_gc(gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.DROP, refs=["HEAD"], dry_run=False)
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is False


# ── ORPHANS ──────────────────────────────────────────────────────────


def test_orphans_never_drops_indexed_commits(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.ORPHANS, refs=[], dry_run=False
    )
    assert counts.commits == 0
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True


def test_orphans_sweeps_crashed_residue(gc: GcFixture) -> None:
    # Simulate a crashed build: snapshot without mark_indexed.
    with gc.store.session() as ws:
        ws.insert_snapshots(
            [Snapshot(commit_sha="crashed", file_path="x.py", blob_sha="bx")],
            repo_id=gc.repo_id,
        )
    run_gc(gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.ORPHANS, refs=[], dry_run=False)
    # Orphan snapshot is gone (swept on session entry or by GC).
    assert gc.store.count_snapshots_for_commit(gc.repo_id, "crashed") == 0
    # Completed commits untouched.
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True


# ── dry-run ──────────────────────────────────────────────────────────


def test_dry_run_reports_without_writing(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=True
    )
    assert counts.commits == 2
    # Nothing actually dropped.
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True


# ── error handling ──────────────────────────────────────────────────


def test_unknown_ref_in_keep_raises(gc: GcFixture) -> None:
    with pytest.raises(RbtrError, match="nosuchref"):
        run_gc(
            gc.store,
            gc.repo.workdir,
            gc.repo_id,
            mode=GcMode.KEEP,
            refs=["nosuchref"],
            dry_run=False,
        )


# ── Search survives GC ───────────────────────────────────────────────


def test_search_works_after_gc(gc: GcFixture) -> None:
    """FTS and get_chunks still work after GC drops commits."""
    run_gc(gc.store, gc.repo.workdir, gc.repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)

    # HEAD (c3) survives — its chunks are still queryable.
    chunks = gc.store.get_chunks(gc.c3, repo_id=gc.repo_id)
    assert len(chunks) > 0

    # Dropped commits return nothing.
    assert gc.store.get_chunks(gc.c1, repo_id=gc.repo_id) == []


# ── Edge cases ───────────────────────────────────────────────────────


def test_gc_preserves_current_worktree_tree_sha(gc: GcFixture) -> None:
    """GC preserves the current worktree tree SHA."""
    # Dirty the working tree to get a real tree SHA.
    workdir = Path(gc.repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("dirty\n")
    tree_sha = worktree_tree_sha(gc.repo.workdir)
    assert tree_sha is not None

    # Re-register with real path and index the tree SHA.
    with gc.store.session() as ws:
        repo_id = ws.register_repo(gc.repo.workdir)
        ws.mark_indexed(repo_id, gc.c3)  # HEAD
        ws.mark_indexed(repo_id, tree_sha)

    run_gc(gc.store, gc.repo.workdir, repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)

    # HEAD and current tree SHA both preserved.
    assert gc.store.has_indexed(repo_id, gc.c3) is True
    assert gc.store.has_indexed(repo_id, tree_sha) is True


def test_gc_drops_stale_worktree_tree_sha(gc: GcFixture) -> None:
    """GC drops stale worktree tree SHAs (not the current one)."""
    workdir = Path(gc.repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs

    # Create a stale tree SHA.
    (workdir / "a.py").write_text("stale\n")
    stale_sha = worktree_tree_sha(gc.repo.workdir)
    assert stale_sha is not None

    # Now edit again to get a different current tree SHA.
    (workdir / "a.py").write_text("current\n")
    current_sha = worktree_tree_sha(gc.repo.workdir)
    assert current_sha is not None
    assert stale_sha != current_sha

    # Register and index both.
    with gc.store.session() as ws:
        repo_id = ws.register_repo(gc.repo.workdir)
        ws.mark_indexed(repo_id, gc.c3)  # HEAD
        ws.mark_indexed(repo_id, stale_sha)
        ws.mark_indexed(repo_id, current_sha)

    run_gc(gc.store, gc.repo.workdir, repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)

    # HEAD and current tree SHA preserved; stale dropped.
    assert gc.store.has_indexed(repo_id, gc.c3) is True
    assert gc.store.has_indexed(repo_id, current_sha) is True
    assert gc.store.has_indexed(repo_id, stale_sha) is False


def test_gc_unborn_head_raises(tmp_path: Path) -> None:
    """GC on a repo with no commits (unborn HEAD) raises RbtrError."""
    repo = pygit2.init_repository(str(tmp_path / "empty"), bare=False, initial_head="main")
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo("/empty")

    with pytest.raises(RbtrError, match="no commits"):
        run_gc(store, repo.workdir, repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False)
    store.close()
