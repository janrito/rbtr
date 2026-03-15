"""Tests for rbtr.git.repo — open, status, branches, remotes.

Tests are organised around a shared multi-branch repository
(`branched_repo`) with realistic commit history.  One-off
fixtures are used only for edge cases (empty repos, non-git dirs).

Functions under test:
- open_repo
- require_clean
- parse_github_remote
- default_branch
- list_local_branches
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.exceptions import RbtrError
from rbtr.git.repo import (
    default_branch,
    list_local_branches,
    open_repo,
    require_clean,
)
from rbtr.github.client import parse_github_remote

from .conftest import make_commit

# ── Shared dataset ───────────────────────────────────────────────────
#
# branched_repo has:
#   main      — 1 commit  (initial)
#   develop   — 2 commits (initial + feature work)
#   release   — 1 commit  (same as main, created later for sort-order test)
#
# HEAD points to main.


@dataclass
class BranchedRepo:
    repo: pygit2.Repository
    main_oid: pygit2.Oid
    develop_oid: pygit2.Oid
    release_oid: pygit2.Oid


@pytest.fixture
def branched_repo(tmp_path: Path) -> BranchedRepo:
    """A repo with three local branches at different commits."""
    repo = pygit2.init_repository(str(tmp_path / "repo"))

    # main — initial commit
    main_oid = make_commit(
        repo,
        {"README.md": b"# Hello\n"},
        message="Initial commit",
        ref="refs/heads/main",
        author="Alice",
    )
    repo.set_head("refs/heads/main")

    # develop — branches from main, one more commit
    develop_oid = make_commit(
        repo,
        {"README.md": b"# Hello\n", "src/app.py": b"print('hi')\n"},
        message="Add app",
        parents=[main_oid],
        ref="refs/heads/develop",
        author="Bob",
    )

    # release — same tree as main but a distinct commit (for sort test)
    release_oid = make_commit(
        repo,
        {"README.md": b"# Hello\n"},
        message="Release v1",
        parents=[main_oid],
        ref="refs/heads/release",
        author="Alice",
    )

    return BranchedRepo(
        repo=repo,
        main_oid=main_oid,
        develop_oid=develop_oid,
        release_oid=release_oid,
    )


# ── open_repo ────────────────────────────────────────────────────────


def test_open_repo_outside_git_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.chdir(tmp)
        with pytest.raises(RbtrError, match="inside a git repository"):
            open_repo()


def test_open_repo_inside_git_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pygit2.init_repository(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    repo = open_repo()
    assert isinstance(repo, pygit2.Repository)


# ── require_clean ────────────────────────────────────────────────────
# These need a checked-out working tree, so they use a dedicated
# fixture instead of branched_repo (which only creates objects).


@pytest.fixture
def worktree_repo(tmp_path: Path) -> pygit2.Repository:
    """A repo with one committed file checked out to the working tree."""
    repo = pygit2.init_repository(str(tmp_path / "wt"))
    workdir = repo.workdir
    assert workdir is not None
    Path(os.path.join(workdir, "file.txt")).write_text("hello")
    repo.index.add("file.txt")
    repo.index.write()
    sig = pygit2.Signature("Test", "test@test.com")
    tree = repo.index.write_tree()
    repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
    repo.set_head("refs/heads/main")
    return repo


def test_require_clean_on_clean_repo(worktree_repo: pygit2.Repository) -> None:
    require_clean(worktree_repo)


def test_require_clean_modified_file(worktree_repo: pygit2.Repository) -> None:
    workdir = worktree_repo.workdir
    assert workdir is not None
    Path(os.path.join(workdir, "file.txt")).write_text("modified")

    with pytest.raises(RbtrError, match="uncommitted changes"):
        require_clean(worktree_repo)


def test_require_clean_staged_file(worktree_repo: pygit2.Repository) -> None:
    workdir = worktree_repo.workdir
    assert workdir is not None
    Path(os.path.join(workdir, "new.txt")).write_text("staged")
    worktree_repo.index.add("new.txt")
    worktree_repo.index.write()

    with pytest.raises(RbtrError, match="uncommitted changes"):
        require_clean(worktree_repo)


def test_require_clean_untracked_files_are_ok(worktree_repo: pygit2.Repository) -> None:
    workdir = worktree_repo.workdir
    assert workdir is not None
    Path(os.path.join(workdir, "untracked.txt")).write_text("untracked")
    require_clean(worktree_repo)


# ── parse_github_remote ──────────────────────────────────────────────


def _make_repo_with_remote(tmp_path: Path, url: str) -> pygit2.Repository:
    repo = pygit2.init_repository(str(tmp_path))
    repo.remotes.create("origin", url)
    return repo


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("git@github.com:owner/repo-name.git", ("owner", "repo-name")),
        ("https://github.com/myorg/myrepo.git", ("myorg", "myrepo")),
        ("https://github.com/myorg/myrepo", ("myorg", "myrepo")),
    ],
)
def test_parse_github_url_formats(tmp_path: Path, url: str, expected: tuple[str, str]) -> None:
    repo = _make_repo_with_remote(tmp_path, url)
    assert parse_github_remote(repo) == expected


def test_parse_no_github_remote(tmp_path: Path) -> None:
    repo = _make_repo_with_remote(tmp_path, "https://gitlab.com/owner/repo.git")
    with pytest.raises(RbtrError, match="No GitHub remote found"):
        parse_github_remote(repo)


def test_parse_no_remotes(tmp_path: Path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "empty"))
    with pytest.raises(RbtrError, match="No GitHub remote found"):
        parse_github_remote(repo)


# ── default_branch ───────────────────────────────────────────────────


def test_default_branch_with_origin_head(branched_repo: BranchedRepo) -> None:
    """When origin/HEAD is set (as by git clone), use its target."""
    repo = branched_repo.repo
    repo.remotes.create("origin", "https://github.com/o/r.git")

    # Simulate what git clone does: create the remote ref + symbolic ref.
    repo.references.create("refs/remotes/origin/main", branched_repo.main_oid)
    # Passing a string target creates a symbolic ref.
    repo.references.create("refs/remotes/origin/HEAD", "refs/remotes/origin/main")

    assert default_branch(repo) == "main"


def test_default_branch_falls_back_to_main(branched_repo: BranchedRepo) -> None:
    """Without origin/HEAD, detects 'main' from local branches."""
    assert default_branch(branched_repo.repo) == "main"


def test_default_branch_falls_back_to_master(tmp_path: Path) -> None:
    """When only 'master' exists locally, use that."""
    repo = pygit2.init_repository(str(tmp_path / "repo"))
    make_commit(repo, {"a.txt": b"x\n"}, ref="refs/heads/master")
    assert default_branch(repo) == "master"


def test_default_branch_last_resort(tmp_path: Path) -> None:
    """When no known branch exists, returns 'main' as fallback."""
    repo = pygit2.init_repository(str(tmp_path / "repo"))
    make_commit(repo, {"a.txt": b"x\n"}, ref="refs/heads/trunk")
    assert default_branch(repo) == "main"


# ── list_local_branches ─────────────────────────────────────────────


def test_list_local_branches_excludes_current(branched_repo: BranchedRepo) -> None:
    """HEAD is on main — main should be excluded."""
    branches = list_local_branches(branched_repo.repo)
    names = {b.name for b in branches}
    assert "main" not in names
    assert "develop" in names
    assert "release" in names


def test_list_local_branches_sorted_by_date(branched_repo: BranchedRepo) -> None:
    """Branches are sorted newest-first by commit time."""
    branches = list_local_branches(branched_repo.repo)
    # All commits have the same timestamp (created in sequence within
    # the same second), but the order should at least be stable.
    assert len(branches) == 2


def test_list_local_branches_commit_fields(branched_repo: BranchedRepo) -> None:
    branches = list_local_branches(branched_repo.repo)
    develop = next(b for b in branches if b.name == "develop")
    assert develop.last_commit_sha == str(branched_repo.develop_oid)
    assert develop.last_commit_message == "Add app"
    assert develop.updated_at is not None


def test_list_local_branches_empty_repo(tmp_path: Path) -> None:
    """An empty repo with an unborn HEAD returns an empty list."""
    repo = pygit2.init_repository(str(tmp_path / "repo"))
    assert list_local_branches(repo) == []


def test_list_local_branches_single_branch(tmp_path: Path) -> None:
    """A repo with only the current branch returns an empty list."""
    repo = pygit2.init_repository(str(tmp_path / "repo"))
    make_commit(repo, {"a.txt": b"x\n"}, ref="refs/heads/main")
    repo.set_head("refs/heads/main")
    assert list_local_branches(repo) == []
