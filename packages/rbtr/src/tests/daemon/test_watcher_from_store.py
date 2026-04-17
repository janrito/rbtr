"""The watcher is seeded from the `repos` table at startup."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pygit2
import pytest

from rbtr.daemon.server import DaemonServer
from rbtr.daemon.watcher import RefWatcher
from rbtr.index.store import IndexStore


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A minimal git repository with one commit."""
    path = tmp_path / "repo"
    path.mkdir()
    pygit2.init_repository(str(path))
    repo = pygit2.Repository(str(path))

    (path / "README.md").write_text("hello\n")
    repo.index.add("README.md")
    repo.index.write()
    tree = repo.index.write_tree()
    sig = pygit2.Signature("tester", "t@t.t")
    repo.create_commit("HEAD", sig, sig, "init", tree, [])
    return path


def test_server_seeds_watcher_from_repos_table(git_repo: Path) -> None:
    store = IndexStore()
    store.register_repo(str(git_repo))

    server = DaemonServer(Path(tempfile.mkdtemp(prefix="rbtr")), store=store)
    # Simulate the serve() startup step without running the event loop
    for _repo_id, path in store.list_repos():
        server._watcher.register(path)

    assert str(git_repo) in server._watcher.repos()


def test_watcher_has_no_persistence_methods() -> None:
    """`save`/`load` were removed in favour of `repos` table seeding."""
    watcher = RefWatcher()
    assert not hasattr(watcher, "save")
    assert not hasattr(watcher, "load")
