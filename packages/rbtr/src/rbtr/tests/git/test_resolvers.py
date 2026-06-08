"""Tests for `rbtr.git.resolve_ref`."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.errors import RbtrError
from rbtr.git import WORKTREE_REF, resolve_ref, worktree_tree_sha

from .conftest import SampleRepo


@pytest.fixture
def sample_repo_path(sample_repo: SampleRepo) -> str:
    """Filesystem path of *sample_repo*'s working directory."""
    return str(Path(sample_repo.repo.path).parent)


# ── Basic resolution ─────────────────────────────────────────────────


@pytest.mark.parametrize("ref", ["HEAD", "main"])
def test_resolve_ref_returns_base_sha(
    sample_repo: SampleRepo, sample_repo_path: str, ref: str
) -> None:
    assert resolve_ref(sample_repo_path, ref) == str(sample_repo.base)


def test_resolve_ref_resolves_branch(sample_repo: SampleRepo, sample_repo_path: str) -> None:
    assert resolve_ref(sample_repo_path, "feature") == str(sample_repo.head)


def test_resolve_ref_resolves_full_sha(sample_repo: SampleRepo, sample_repo_path: str) -> None:
    sha = str(sample_repo.head)
    assert resolve_ref(sample_repo_path, sha) == sha


def test_resolve_ref_raises_on_unknown(sample_repo_path: str) -> None:
    with pytest.raises(RbtrError, match="no_such_branch"):
        resolve_ref(sample_repo_path, "no_such_branch")


def test_resolve_ref_raises_on_missing_repo(tmp_path: Path) -> None:
    bogus = str(tmp_path / "no-such-repo")
    with pytest.raises(RbtrError):
        resolve_ref(bogus, "HEAD")


# ── SHA short-circuit ────────────────────────────────────────────────


def test_sha_short_circuits_without_repo(tmp_path: Path) -> None:
    sha = "deadbeef" * 5  # 40 hex chars
    bogus = str(tmp_path / "no-such-repo")
    assert resolve_ref(bogus, sha) == sha


# ── Worktree symbolic ref ─────────────────────────────────────────────


@pytest.fixture
def worktree_repo(tmp_path: Path) -> pygit2.Repository:
    """Non-bare repo for worktree ref tests."""
    repo = pygit2.init_repository(str(tmp_path / "wt"), bare=False, initial_head="main")
    workdir = Path(repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("def a(): pass\n")
    idx = repo.index
    idx.add("a.py")
    idx.write()
    tree = idx.write_tree()
    sig = pygit2.Signature("T", "t@t")
    repo.create_commit("HEAD", sig, sig, "init", tree, [])
    return repo


def test_resolve_ref_worktree_dirty(worktree_repo: pygit2.Repository) -> None:
    """resolve_ref(WORKTREE_REF) returns the tree SHA when dirty."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("edited\n")

    sha = resolve_ref(worktree_repo.workdir, WORKTREE_REF)
    expected = worktree_tree_sha(worktree_repo.workdir)
    assert sha == expected
    assert len(sha) == 40


def test_resolve_ref_worktree_clean_raises(worktree_repo: pygit2.Repository) -> None:
    """resolve_ref(WORKTREE_REF) raises when the tree is clean."""
    with pytest.raises(RbtrError, match="clean"):
        resolve_ref(worktree_repo.workdir, WORKTREE_REF)


def test_resolve_ref_raw_tree_sha(worktree_repo: pygit2.Repository) -> None:
    """resolve_ref with a raw tree SHA returns it as-is."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("edited\n")

    tree_sha = worktree_tree_sha(worktree_repo.workdir)
    assert tree_sha is not None
    assert resolve_ref(worktree_repo.workdir, tree_sha) == tree_sha
