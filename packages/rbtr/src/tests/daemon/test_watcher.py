"""Tests for the ref watcher.

The watcher is a single stateless function: given an `IndexStore`,
it iterates every registered repo and reports any whose current HEAD
has not been marked as fully indexed. The tests exercise this
function directly against a real `IndexStore` and real git repos,
which keeps them honest — there is no `RefWatcher` class to mock.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.watcher import StaleHead, poll
from rbtr.index.store import IndexStore


def _make_commit(repo: pygit2.Repository, files: dict[str, bytes]) -> pygit2.Oid:
    tb = repo.TreeBuilder()
    for name, content in files.items():
        blob_id = repo.create_blob(content)
        tb.insert(name, blob_id, pygit2.GIT_FILEMODE_BLOB)
    tree_id = tb.write()
    sig = pygit2.Signature("test", "test@test.com")
    parents = [repo.head.target] if not repo.head_is_unborn else []
    return repo.create_commit("refs/heads/main", sig, sig, "commit", tree_id, parents)


@pytest.fixture
def git_repo(tmp_path: Path) -> pygit2.Repository:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=False)
    _make_commit(repo, {"a.py": b"x = 1\n"})
    return repo


@pytest.fixture
def store() -> IndexStore:
    return IndexStore()


def _workdir(repo: pygit2.Repository) -> str:
    assert repo.workdir is not None
    return repo.workdir


def test_unregistered_repo_is_ignored(store: IndexStore) -> None:
    """Repos not in the store are not polled."""
    assert poll(store) == []


def test_indexed_head_yields_no_stale(
    git_repo: pygit2.Repository, store: IndexStore
) -> None:
    path = _workdir(git_repo)
    repo_id = store.register_repo(path)
    store.mark_indexed(repo_id, str(git_repo.head.target))

    assert poll(store) == []


def test_unindexed_head_is_stale(
    git_repo: pygit2.Repository, store: IndexStore
) -> None:
    """A registered repo whose HEAD has no indexed_commits row is stale."""
    path = _workdir(git_repo)
    store.register_repo(path)

    out = poll(store)

    assert out == [StaleHead(repo_path=path, new_ref=str(git_repo.head.target))]


def test_new_commit_after_indexing_is_stale(
    git_repo: pygit2.Repository, store: IndexStore
) -> None:
    """Indexing an older commit doesn't mark a new HEAD fresh."""
    path = _workdir(git_repo)
    repo_id = store.register_repo(path)
    old_head = str(git_repo.head.target)
    store.mark_indexed(repo_id, old_head)

    _make_commit(git_repo, {"b.py": b"y = 2\n"})
    new_head = str(git_repo.head.target)

    out = poll(store)
    assert out == [StaleHead(repo_path=path, new_ref=new_head)]
    assert new_head != old_head


def test_missing_path_is_skipped(store: IndexStore, tmp_path: Path) -> None:
    """A registered path that is not a git repo is silently skipped."""
    store.register_repo(str(tmp_path / "does-not-exist"))
    assert poll(store) == []


def test_multiple_repos_reports_only_stale(
    tmp_path: Path, store: IndexStore
) -> None:
    repo_a = pygit2.init_repository(str(tmp_path / "a"), bare=False)
    _make_commit(repo_a, {"a.py": b"1\n"})
    repo_b = pygit2.init_repository(str(tmp_path / "b"), bare=False)
    _make_commit(repo_b, {"b.py": b"2\n"})

    path_a = _workdir(repo_a)
    path_b = _workdir(repo_b)
    id_a = store.register_repo(path_a)
    store.register_repo(path_b)

    # A is indexed; B is not.
    store.mark_indexed(id_a, str(repo_a.head.target))

    out = poll(store)
    assert out == [StaleHead(repo_path=path_b, new_ref=str(repo_b.head.target))]
