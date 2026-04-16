"""Tests for the ref watcher — detects HEAD changes in repos."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.watcher import RefWatcher


@pytest.fixture
def git_repo(tmp_path: Path) -> pygit2.Repository:
    """A git repo with one commit."""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=False)
    _make_commit(repo, {"a.py": b"x = 1\n"})
    return repo


def _make_commit(repo: pygit2.Repository, files: dict[str, bytes]) -> pygit2.Oid:
    """Create a commit with the given files."""
    tb = repo.TreeBuilder()
    for name, content in files.items():
        blob_id = repo.create_blob(content)
        tb.insert(name, blob_id, pygit2.GIT_FILEMODE_BLOB)
    tree_id = tb.write()
    sig = pygit2.Signature("test", "test@test.com")
    parents = [repo.head.target] if not repo.head_is_unborn else []
    return repo.create_commit("refs/heads/main", sig, sig, "commit", tree_id, parents)


def test_no_change_after_initial_poll(git_repo: pygit2.Repository) -> None:
    watcher = RefWatcher()
    repo_path = str(Path(git_repo.workdir).resolve())
    watcher.register(repo_path)
    changes = watcher.poll()
    assert changes == []


def test_detects_new_commit(git_repo: pygit2.Repository) -> None:
    watcher = RefWatcher()
    repo_path = str(Path(git_repo.workdir).resolve())
    watcher.register(repo_path)
    watcher.poll()  # snapshot current HEAD

    _make_commit(git_repo, {"b.py": b"y = 2\n"})

    changes = watcher.poll()
    assert len(changes) == 1
    assert changes[0].repo_path == repo_path
    assert changes[0].old_ref != changes[0].new_ref


def test_no_change_without_commit(git_repo: pygit2.Repository) -> None:
    watcher = RefWatcher()
    repo_path = str(Path(git_repo.workdir).resolve())
    watcher.register(repo_path)
    watcher.poll()
    changes = watcher.poll()
    assert changes == []


def test_unregister(git_repo: pygit2.Repository) -> None:
    watcher = RefWatcher()
    repo_path = str(Path(git_repo.workdir).resolve())
    watcher.register(repo_path)
    watcher.unregister(repo_path)
    _make_commit(git_repo, {"c.py": b"z = 3\n"})
    changes = watcher.poll()
    assert changes == []


def test_multiple_repos(tmp_path: Path) -> None:
    repo_a = pygit2.init_repository(str(tmp_path / "a"), bare=False)
    repo_b = pygit2.init_repository(str(tmp_path / "b"), bare=False)
    _make_commit(repo_a, {"a.py": b"1\n"})
    _make_commit(repo_b, {"b.py": b"2\n"})

    watcher = RefWatcher()
    path_a = str(Path(repo_a.workdir).resolve())
    path_b = str(Path(repo_b.workdir).resolve())
    watcher.register(path_a)
    watcher.register(path_b)
    watcher.poll()  # snapshot

    _make_commit(repo_a, {"a2.py": b"3\n"})
    # repo_b unchanged

    changes = watcher.poll()
    assert len(changes) == 1
    assert changes[0].repo_path == path_a
