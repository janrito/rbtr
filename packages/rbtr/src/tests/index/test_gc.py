"""Tests for ``rbtr.index.gc.run_gc``.

Covers every mode against a tiny seeded repo with three commits.
Uses real pygit2 repositories and a real in-memory IndexStore so
we exercise the actual code path, not mocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.messages import GcMode
from rbtr.errors import RbtrError
from rbtr.index.gc import run_gc
from rbtr.index.models import Chunk, ChunkKind
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
def gc_repo(tmp_path: Path, gc_signature: pygit2.Signature) -> tuple[pygit2.Repository, str, str, str]:
    """Three-commit repo.  Tags ``v1`` at the first commit so KEEP_REFS has
    a non-HEAD ref to preserve; leaves the middle commit unreachable so
    HEAD_ONLY has something to drop."""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=False)
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
) -> GcFixture:
    """Index all three commits into a fresh in-memory store."""
    repo, c1, c2, c3 = gc_repo
    store = IndexStore()
    repo_id = store.register_repo("/repo")
    for i, sha in enumerate((c1, c2, c3)):
        chunk = Chunk(
            id=f"c{i}",
            blob_sha=f"b{i}",
            file_path="a.py",
            kind=ChunkKind.FUNCTION,
            name="f",
            content="",
            line_start=1,
            line_end=1,
        )
        store.insert_chunks([chunk], repo_id=repo_id)
        store.insert_snapshot(sha, "a.py", f"b{i}", repo_id=repo_id)
        store.mark_indexed(repo_id, sha)
    return GcFixture(repo=repo, store=store, repo_id=repo_id, c1=c1, c2=c2, c3=c3)


# ── HEAD_ONLY ────────────────────────────────────────────────────────


def test_head_only_keeps_head_and_drops_rest(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo, gc.repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False
    )
    assert counts.commits == 2  # c1 and c2 dropped
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is False
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


# ── KEEP_REFS ────────────────────────────────────────────────────────


def test_keep_refs_preserves_tag_and_head(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo, gc.repo_id, mode=GcMode.KEEP_REFS, refs=[], dry_run=False
    )
    assert counts.commits == 1  # only c2 (unreachable) dropped
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True  # tag v1
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True  # HEAD
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


# ── KEEP ─────────────────────────────────────────────────────────────


def test_keep_preserves_listed_refs_and_head(gc: GcFixture) -> None:
    # Keep v1 (c1). HEAD (c3) is kept implicitly.
    counts = run_gc(
        gc.store, gc.repo, gc.repo_id, mode=GcMode.KEEP, refs=["v1"], dry_run=False
    )
    assert counts.commits == 1
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


def test_keep_with_no_refs_is_head_only(gc: GcFixture) -> None:
    run_gc(
        gc.store, gc.repo, gc.repo_id, mode=GcMode.KEEP, refs=[], dry_run=False
    )
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True  # HEAD kept
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is False
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is False


# ── DROP ─────────────────────────────────────────────────────────────


def test_drop_removes_only_listed_refs(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo, gc.repo_id, mode=GcMode.DROP, refs=["v1"], dry_run=False
    )
    assert counts.commits == 1
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is False
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is True


def test_drop_head_is_allowed(gc: GcFixture) -> None:
    """DROP does not implicitly keep HEAD — if the user says drop HEAD, do it."""
    run_gc(gc.store, gc.repo, gc.repo_id, mode=GcMode.DROP, refs=["HEAD"], dry_run=False)
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is False


# ── ORPHANS ──────────────────────────────────────────────────────────


def test_orphans_never_drops_indexed_commits(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo, gc.repo_id, mode=GcMode.ORPHANS, refs=[], dry_run=False
    )
    assert counts.commits == 0
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c2) is True
    assert gc.store.has_indexed(gc.repo_id, gc.c3) is True


def test_orphans_sweeps_crashed_residue(gc: GcFixture) -> None:
    # Simulate a crashed build: snapshot without mark_indexed.
    gc.store.insert_snapshot("crashed", "x.py", "bx", repo_id=gc.repo_id)
    counts = run_gc(
        gc.store, gc.repo, gc.repo_id, mode=GcMode.ORPHANS, refs=[], dry_run=False
    )
    assert counts.snapshots == 1
    # Completed commits untouched.
    assert gc.store.has_indexed(gc.repo_id, gc.c1) is True


# ── dry-run ──────────────────────────────────────────────────────────


def test_dry_run_reports_without_writing(gc: GcFixture) -> None:
    counts = run_gc(
        gc.store, gc.repo, gc.repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=True
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
            gc.repo,
            gc.repo_id,
            mode=GcMode.KEEP,
            refs=["nosuchref"],
            dry_run=False,
        )
