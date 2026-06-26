"""End-to-end smoke for `rbtr index --remove-stale-refs` via subprocess.

Exercises the inline (no-daemon) prune path: watched refs that no
longer resolve are removed; HEAD and resolvable refs are kept.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.git import normalise_repo_path
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


def test_fresh_repo_indexes_end_to_end(git_repo: pygit2.Repository, isolated_db: Path) -> None:
    """A fresh repo indexes end-to-end via the real CLI.

    Resilience acceptance for the Aim: opening a new repo gets it
    indexed.  `--no-daemon --no-embed` runs the build inline and
    loads no embedding model — GPU-free, and no daemon to contend
    with the rest of the suite (the daemon-start race is covered by
    `test_start_concurrency`).
    """
    repo = str(git_repo.workdir)
    result = run_cli(["index", "--no-daemon", "--no-embed", "--repo-path", repo])
    assert result.returncode == 0, result.stderr

    store = IndexStore.from_config(writable=True)
    try:
        repo_id = store.get_repo_id(normalise_repo_path(repo))
        assert repo_id is not None, "repo not registered"
        commits = store.list_indexed_commits(repo_id)
        assert len(commits) == 1, "HEAD not indexed"
        assert store.count_chunks(commits[0][0], repo_id) > 0, "no symbols extracted"
    finally:
        store.close()


def test_index_remove_stale_refs_prunes_unresolvable(repo_with_stale_watch: str) -> None:
    r = run_cli(
        ["index", "--remove-stale-refs", "--no-daemon", "--repo-path", repo_with_stale_watch]
    )
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


@pytest.fixture
def head_only_repo(tmp_path: Path, isolated_db: Path) -> str:
    """A real repo registered with HEAD as its only watched ref."""
    path = tmp_path / "solo"
    repo = pygit2.init_repository(str(path), bare=False, initial_head="main")
    sig = pygit2.Signature("t", "t@t.t")
    repo.create_commit("refs/heads/main", sig, sig, "init", repo.TreeBuilder().write(), [])
    resolved = normalise_repo_path(str(path))
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(resolved)
        ws.add_watched_refs(repo_id, ["HEAD"])
    store.close()
    return str(path)


def test_index_remove_no_refs_forgets_head_only_repo(head_only_repo: str) -> None:
    """`rbtr index --remove` with no refs forgets a HEAD-only repo."""
    r = run_cli(["index", "--remove", "--no-daemon", "--repo-path", head_only_repo])
    assert r.returncode == 0, r.stderr
    store = IndexStore.from_config(writable=True)
    try:
        assert store.get_repo_id(normalise_repo_path(head_only_repo)) is None
    finally:
        store.close()


def test_index_remove_stale_repos_forgets_vanished(tmp_path: Path, isolated_db: Path) -> None:
    """`--remove-stale-repos` forgets a repo whose path is gone, needing no
    current repo of its own."""
    gone = str(tmp_path / "gone")  # never created on disk
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        ws.register_repo(gone)
    store.close()

    r = run_cli(["index", "--remove-stale-repos", "--no-daemon"])
    assert r.returncode == 0, r.stderr

    store = IndexStore.from_config(writable=True)
    try:
        assert store.get_repo_id(gone) is None
    finally:
        store.close()
