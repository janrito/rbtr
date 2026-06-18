"""End-to-end smoke for `rbtr index --remove-stale` via subprocess.

Exercises the inline (no-daemon) prune path: watched refs that no
longer resolve are removed; HEAD and resolvable refs are kept.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.index.store import IndexStore
from rbtr.tests.conftest import run_cli


@pytest.fixture
def repo_with_stale_watch(tmp_path: Path, isolated_db: Path) -> str:
    """A real repo whose watch set holds HEAD, main, and a deleted branch."""
    path = tmp_path / "repo"
    repo = pygit2.init_repository(str(path), bare=False, initial_head="main")
    sig = pygit2.Signature("t", "t@t.t")
    repo.create_commit("refs/heads/main", sig, sig, "init", repo.TreeBuilder().write(), [])
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(str(path))
        ws.add_watched_refs(repo_id, ["HEAD", "main", "gone-branch"])
    store.close()
    return str(path)


def test_index_remove_stale_prunes_unresolvable(repo_with_stale_watch: str) -> None:
    r = run_cli(["index", "--remove-stale", "--no-daemon", "--repo-path", repo_with_stale_watch])
    assert r.returncode == 0, r.stderr

    store = IndexStore.from_config(writable=True)
    try:
        repo_id = store.get_repo_id(repo_with_stale_watch)
        assert repo_id is not None
        watched = store.list_watched_refs(repo_id)
    finally:
        store.close()
    assert "gone-branch" not in watched  # unresolvable → pruned
    assert "main" in watched  # resolvable → kept
    assert "HEAD" in watched  # always kept
