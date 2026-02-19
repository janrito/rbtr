"""Tests for rbtr.repo."""

import os
import tempfile

import pygit2
import pytest

from rbtr.exceptions import RbtrError
from rbtr.repo import open_repo, parse_github_remote, require_clean

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def git_repo(tmp_path):
    """Create a git repo with one commit on main."""
    repo = pygit2.init_repository(str(tmp_path))
    sig = pygit2.Signature("Test", "test@test.com")
    tree = repo.TreeBuilder().write()
    repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
    repo.set_head("refs/heads/main")
    return repo


# ── open_repo ────────────────────────────────────────────────────────


def test_open_repo_outside_git_repo(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.chdir(tmp)
        with pytest.raises(RbtrError, match="inside a git repository"):
            open_repo()


def test_open_repo_inside_git_repo(monkeypatch, tmp_path) -> None:
    pygit2.init_repository(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    repo = open_repo()
    assert isinstance(repo, pygit2.Repository)


# ── require_clean ────────────────────────────────────────────────────


def test_require_clean_on_clean_repo(git_repo) -> None:
    require_clean(git_repo)


def test_require_clean_modified_file(git_repo) -> None:
    workdir = git_repo.workdir
    filepath = os.path.join(workdir, "file.txt")
    with open(filepath, "w") as f:
        f.write("hello")
    git_repo.index.add("file.txt")
    git_repo.index.write()
    sig = pygit2.Signature("Test", "test@test.com")
    tree = git_repo.index.write_tree()
    parent = git_repo.head.peel(pygit2.Commit)
    git_repo.create_commit("refs/heads/main", sig, sig, "add file", tree, [parent.id])

    with open(filepath, "w") as f:
        f.write("modified")

    with pytest.raises(RbtrError, match="uncommitted changes"):
        require_clean(git_repo)


def test_require_clean_staged_file(git_repo) -> None:
    workdir = git_repo.workdir
    filepath = os.path.join(workdir, "new.txt")
    with open(filepath, "w") as f:
        f.write("staged")
    git_repo.index.add("new.txt")
    git_repo.index.write()

    with pytest.raises(RbtrError, match="uncommitted changes"):
        require_clean(git_repo)


def test_require_clean_untracked_files_are_ok(git_repo) -> None:
    workdir = git_repo.workdir
    filepath = os.path.join(workdir, "untracked.txt")
    with open(filepath, "w") as f:
        f.write("untracked")
    require_clean(git_repo)


# ── parse_github_remote ──────────────────────────────────────────────


def _make_repo_with_remote(tmp_path, url: str) -> pygit2.Repository:
    repo = pygit2.init_repository(str(tmp_path))
    repo.remotes.create("origin", url)
    return repo


def test_parse_ssh_url(tmp_path) -> None:
    repo = _make_repo_with_remote(tmp_path, "git@github.com:owner/repo-name.git")
    assert parse_github_remote(repo) == ("owner", "repo-name")


def test_parse_https_url(tmp_path) -> None:
    repo = _make_repo_with_remote(tmp_path, "https://github.com/myorg/myrepo.git")
    assert parse_github_remote(repo) == ("myorg", "myrepo")


def test_parse_https_without_dotgit(tmp_path) -> None:
    repo = _make_repo_with_remote(tmp_path, "https://github.com/myorg/myrepo")
    assert parse_github_remote(repo) == ("myorg", "myrepo")


def test_parse_no_github_remote(tmp_path) -> None:
    repo = _make_repo_with_remote(tmp_path, "https://gitlab.com/owner/repo.git")
    with pytest.raises(RbtrError, match="No GitHub remote found"):
        parse_github_remote(repo)


def test_parse_no_remotes(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path))
    with pytest.raises(RbtrError, match="No GitHub remote found"):
        parse_github_remote(repo)
