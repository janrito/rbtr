"""Tests for `_resolve_read_ref` routing logic.

Verifies the ref behaviours:
- `None` → worktree tree SHA if dirty and indexed, HEAD otherwise
- `"HEAD"` → always committed state
- explicit ref → resolved literally
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pygit2
from pytest_cases import fixture, parametrize_with_cases

from rbtr.daemon.handlers import _resolve_read_ref
from rbtr.git import worktree_tree_sha
from rbtr.index.store import IndexStore

from .cases_resolve_ref import RefScenario


@fixture
def ref_sig() -> pygit2.Signature:
    return pygit2.Signature("t", "t@t.t")


@fixture
@parametrize_with_cases("s")
def ref_scenario(s: RefScenario) -> RefScenario:
    return s


@fixture
def ref_repo(
    ref_scenario: RefScenario,
    tmp_path: Path,
    ref_sig: pygit2.Signature,
) -> pygit2.Repository:
    """Repo with two branches: main (HEAD) and feature.

    Optionally dirties a tracked file per the scenario.
    """
    repo_dir = tmp_path / "ref_repo"
    repo_dir.mkdir()
    repo = pygit2.init_repository(str(repo_dir), bare=False, initial_head="main")
    # Write a.py to disk and commit it properly.
    (repo_dir / "a.py").write_text("x = 1\n")
    repo.index.add("a.py")
    repo.index.write()
    tree = repo.index.write_tree()
    head_sha = repo.create_commit("refs/heads/main", ref_sig, ref_sig, "init", tree, [])
    # Feature branch with a second commit.
    tb2 = repo.TreeBuilder()
    tb2.insert("a.py", repo.create_blob(b"x = 2\n"), pygit2.GIT_FILEMODE_BLOB)
    repo.create_commit("refs/heads/feature", ref_sig, ref_sig, "feat", tb2.write(), [head_sha])
    # Dirty the file if the scenario asks.
    if ref_scenario.dirty_worktree:
        (repo_dir / "a.py").write_text("x = dirty\n")
    return repo


@fixture
def ref_store(
    ref_scenario: RefScenario,
    ref_repo: pygit2.Repository,
) -> Generator[IndexStore]:
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(ref_repo.workdir)
        ws.mark_indexed(repo_id, str(ref_repo.head.target))
        if ref_scenario.tree_sha_indexed:
            tree_sha = worktree_tree_sha(ref_repo.workdir)
            if tree_sha is not None:
                ws.mark_indexed(repo_id, tree_sha)
    yield store
    store.close()


def test_resolve_read_ref(
    ref_scenario: RefScenario,
    ref_repo: pygit2.Repository,
    ref_store: IndexStore,
) -> None:
    repo_id = ref_store.resolve_repo(ref_repo.workdir)
    result = _resolve_read_ref(ref_store, ref_repo.workdir, repo_id, ref_scenario.requested_ref)

    # Map symbolic names to actual SHAs.
    head_sha = str(ref_repo.head.target)
    feature_sha = str(ref_repo.references["refs/heads/feature"].target)
    tree_sha = worktree_tree_sha(ref_repo.workdir)
    expected_map: dict[str, str | None] = {
        "HEAD_SHA": head_sha,
        "FEATURE_SHA": feature_sha,
        "TREE_SHA": tree_sha,
    }
    assert result == expected_map[ref_scenario.expected]
