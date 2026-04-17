"""Tests for ``rbtr.index.gc.run_gc``.

Covers every mode against a tiny seeded repo with three commits.
Uses real pygit2 repositories and a real in-memory IndexStore so
we exercise the actual code path, not mocks.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.messages import GcMode
from rbtr.errors import RbtrError
from rbtr.index.gc import run_gc
from rbtr.index.models import Chunk, ChunkKind
from rbtr.index.store import IndexStore


@pytest.fixture
def store() -> IndexStore:
    return IndexStore()


def _commit(repo: pygit2.Repository, files: dict[str, bytes], ref: str) -> pygit2.Oid:
    tb = repo.TreeBuilder()
    for name, content in files.items():
        tb.insert(name, repo.create_blob(content), pygit2.GIT_FILEMODE_BLOB)
    sig = pygit2.Signature("t", "t@t.t")
    parents = [repo.head.target] if not repo.head_is_unborn else []
    return repo.create_commit(ref, sig, sig, "c", tb.write(), parents)


@pytest.fixture
def repo_and_shas(tmp_path: Path) -> tuple[pygit2.Repository, str, str, str]:
    """A repo with HEAD on main, a tag at old_sha, and a dangling sha."""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=False)
    c1 = _commit(repo, {"a.py": b"x = 1\n"}, "refs/heads/main")
    c2 = _commit(repo, {"a.py": b"x = 2\n"}, "refs/heads/main")
    c3 = _commit(repo, {"a.py": b"x = 3\n"}, "refs/heads/main")
    repo.create_reference("refs/tags/v1", c1)
    # c2 is no longer reachable by a named ref; still indexed.
    return repo, str(c1), str(c2), str(c3)


def _seed(store: IndexStore, *shas: str) -> int:
    repo_id = store.register_repo("/repo")
    for i, sha in enumerate(shas):
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
    return repo_id


# ── HEAD_ONLY ────────────────────────────────────────────────────────


def test_head_only_keeps_head_and_drops_rest(
    store: IndexStore, repo_and_shas: tuple[pygit2.Repository, str, str, str]
) -> None:
    repo, c1, c2, c3 = repo_and_shas
    repo_id = _seed(store, c1, c2, c3)

    counts = run_gc(
        store, repo, repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=False
    )

    assert counts.commits == 2  # c1 and c2 dropped
    assert store.has_indexed(repo_id, c3) is True
    assert store.has_indexed(repo_id, c1) is False
    assert store.has_indexed(repo_id, c2) is False


# ── KEEP_REFS ────────────────────────────────────────────────────────


def test_keep_refs_preserves_tag_and_head(
    store: IndexStore, repo_and_shas: tuple[pygit2.Repository, str, str, str]
) -> None:
    repo, c1, c2, c3 = repo_and_shas
    repo_id = _seed(store, c1, c2, c3)

    counts = run_gc(
        store, repo, repo_id, mode=GcMode.KEEP_REFS, refs=[], dry_run=False
    )

    assert counts.commits == 1  # only c2 (unreachable) dropped
    assert store.has_indexed(repo_id, c1) is True  # tag v1
    assert store.has_indexed(repo_id, c3) is True  # HEAD
    assert store.has_indexed(repo_id, c2) is False


# ── KEEP ─────────────────────────────────────────────────────────────


def test_keep_preserves_listed_refs_and_head(
    store: IndexStore, repo_and_shas: tuple[pygit2.Repository, str, str, str]
) -> None:
    repo, c1, c2, c3 = repo_and_shas
    repo_id = _seed(store, c1, c2, c3)

    # Keep v1 (c1). HEAD (c3) is kept implicitly.
    counts = run_gc(
        store, repo, repo_id, mode=GcMode.KEEP, refs=["v1"], dry_run=False
    )

    assert counts.commits == 1
    assert store.has_indexed(repo_id, c1) is True
    assert store.has_indexed(repo_id, c3) is True
    assert store.has_indexed(repo_id, c2) is False


def test_keep_with_no_refs_is_head_only(
    store: IndexStore, repo_and_shas: tuple[pygit2.Repository, str, str, str]
) -> None:
    repo, c1, c2, c3 = repo_and_shas
    repo_id = _seed(store, c1, c2, c3)

    run_gc(
        store, repo, repo_id, mode=GcMode.KEEP, refs=[], dry_run=False
    )

    assert store.has_indexed(repo_id, c3) is True  # HEAD kept
    assert store.has_indexed(repo_id, c1) is False
    assert store.has_indexed(repo_id, c2) is False


# ── DROP ─────────────────────────────────────────────────────────────


def test_drop_removes_only_listed_refs(
    store: IndexStore, repo_and_shas: tuple[pygit2.Repository, str, str, str]
) -> None:
    repo, c1, c2, c3 = repo_and_shas
    repo_id = _seed(store, c1, c2, c3)

    counts = run_gc(
        store, repo, repo_id, mode=GcMode.DROP, refs=["v1"], dry_run=False
    )

    assert counts.commits == 1
    assert store.has_indexed(repo_id, c1) is False
    assert store.has_indexed(repo_id, c3) is True
    assert store.has_indexed(repo_id, c2) is True


def test_drop_head_is_allowed(
    store: IndexStore, repo_and_shas: tuple[pygit2.Repository, str, str, str]
) -> None:
    """DROP does not implicitly keep HEAD — if the user says drop HEAD, do it."""
    repo, c1, c2, c3 = repo_and_shas
    repo_id = _seed(store, c1, c2, c3)

    run_gc(store, repo, repo_id, mode=GcMode.DROP, refs=["HEAD"], dry_run=False)

    assert store.has_indexed(repo_id, c3) is False


# ── ORPHANS ──────────────────────────────────────────────────────────


def test_orphans_never_drops_indexed_commits(
    store: IndexStore, repo_and_shas: tuple[pygit2.Repository, str, str, str]
) -> None:
    repo, c1, c2, c3 = repo_and_shas
    repo_id = _seed(store, c1, c2, c3)

    counts = run_gc(
        store, repo, repo_id, mode=GcMode.ORPHANS, refs=[], dry_run=False
    )

    assert counts.commits == 0
    assert store.has_indexed(repo_id, c1) is True
    assert store.has_indexed(repo_id, c2) is True
    assert store.has_indexed(repo_id, c3) is True


def test_orphans_sweeps_crashed_residue(
    store: IndexStore, repo_and_shas: tuple[pygit2.Repository, str, str, str]
) -> None:
    repo, c1, _c2, _c3 = repo_and_shas
    repo_id = _seed(store, c1)
    # Simulate crashed build: snapshot without mark_indexed.
    store.insert_snapshot("crashed", "x.py", "bx", repo_id=repo_id)

    counts = run_gc(
        store, repo, repo_id, mode=GcMode.ORPHANS, refs=[], dry_run=False
    )

    assert counts.snapshots == 1
    # c1 still present.
    assert store.has_indexed(repo_id, c1) is True


# ── dry-run ──────────────────────────────────────────────────────────


def test_dry_run_reports_without_writing(
    store: IndexStore, repo_and_shas: tuple[pygit2.Repository, str, str, str]
) -> None:
    repo, c1, c2, c3 = repo_and_shas
    repo_id = _seed(store, c1, c2, c3)

    counts = run_gc(
        store, repo, repo_id, mode=GcMode.HEAD_ONLY, refs=[], dry_run=True
    )

    assert counts.commits == 2
    # Nothing actually dropped.
    assert store.has_indexed(repo_id, c1) is True
    assert store.has_indexed(repo_id, c2) is True
    assert store.has_indexed(repo_id, c3) is True


# ── error handling ──────────────────────────────────────────────────


def test_unknown_ref_in_keep_raises(
    store: IndexStore, repo_and_shas: tuple[pygit2.Repository, str, str, str]
) -> None:
    repo, c1, _c2, _c3 = repo_and_shas
    repo_id = _seed(store, c1)

    with pytest.raises(RbtrError, match="nosuchref"):
        run_gc(
            store,
            repo,
            repo_id,
            mode=GcMode.KEEP,
            refs=["nosuchref"],
            dry_run=False,
        )
