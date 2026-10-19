"""Tests for `resolve_read_ref` routing logic.

Verifies the ref behaviours:
- `None` → worktree tree SHA if dirty and indexed, HEAD otherwise
- `"HEAD"` → always committed state
- explicit ref → resolved literally
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.config import config
from rbtr.domain.models import RefSource
from rbtr.errors import IndexNotBuiltError
from rbtr.git import worktree_tree_sha
from rbtr.index.index import Index
from rbtr.index.refs import resolve_read_ref

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
def ref_index(
    ref_scenario: RefScenario,
    ref_repo: pygit2.Repository,
    index: Index,
) -> Index:
    with index.session() as ws:
        repo = ws.register_repo(ref_repo.workdir)
        ws.mark_indexed(at=repo.at(str(ref_repo.head.target)))
        if ref_scenario.tree_sha_indexed:
            tree_sha = worktree_tree_sha(
                ref_repo.workdir, max_untracked_files=config.max_untracked_files
            )
            if tree_sha is not None:
                ws.mark_indexed(at=repo.at(tree_sha))
    return index


def test_resolve_read_ref(
    ref_scenario: RefScenario,
    ref_repo: pygit2.Repository,
    ref_index: Index,
) -> None:
    repo = ref_index.repo(ref_repo.workdir)
    result = resolve_read_ref(ref_index, repo, ref_scenario.requested_ref)

    # Map symbolic names to actual SHAs.
    head_sha = str(ref_repo.head.target)
    feature_sha = str(ref_repo.references["refs/heads/feature"].target)
    tree_sha = worktree_tree_sha(ref_repo.workdir, max_untracked_files=config.max_untracked_files)
    expected_map: dict[str, str | None] = {
        "HEAD_SHA": head_sha,
        "FEATURE_SHA": feature_sha,
        "TREE_SHA": tree_sha,
    }
    assert result.at.snapshot_sha == expected_map[ref_scenario.expected]
    assert result.source == ref_scenario.expected_source


#
# These exercise the `require_indexed` path directly (the case matrix
# above always indexes HEAD). A clean one-commit repo is enough: its
# worktree tree SHA is never indexed here, so resolution always lands
# on HEAD.


def _gate_repo(tmp_path: Path, sig: pygit2.Signature) -> pygit2.Repository:
    repo_dir = tmp_path / "gate_repo"
    repo_dir.mkdir()
    repo = pygit2.init_repository(str(repo_dir), bare=False, initial_head="main")
    (repo_dir / "a.py").write_text("x = 1\n")
    repo.index.add("a.py")
    repo.index.write()
    tree = repo.index.write_tree()
    repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
    return repo


def test_implicit_unindexed_head_falls_back_to_latest_indexed(
    tmp_path: Path, ref_sig: pygit2.Signature, index: Index
) -> None:
    """Implicit resolution prefers an older indexed commit over an
    unindexed HEAD rather than returning an empty result."""
    repo = _gate_repo(tmp_path, ref_sig)
    older = "a" * 40
    with index.session() as ws:
        registered = ws.register_repo(repo.workdir)
        ws.mark_indexed(at=registered.at(older))

    result = resolve_read_ref(index, registered, None, require_indexed=True)

    assert result.at == registered.at(older)
    assert result.source == RefSource.LATEST_INDEXED


def test_implicit_unindexed_head_nothing_indexed_errors(
    tmp_path: Path, ref_sig: pygit2.Signature, index: Index
) -> None:
    """With nothing indexed, the implicit path errors clearly instead of
    silently resolving to an unqueryable ref.

    Raises `IndexNotBuiltError` (the daemon's `_dispatch` upgrades it to
    a 'building' message when a build is active); the wording is the
    plain not-indexed guidance here.
    """
    repo = _gate_repo(tmp_path, ref_sig)
    with index.session() as ws:
        registered = ws.register_repo(repo.workdir)

    with pytest.raises(IndexNotBuiltError, match="not indexed"):
        resolve_read_ref(index, registered, None, require_indexed=True)
