"""Tests for `IndexStore.diff_symbols` — symbol-level diffing.

The diff is computed entirely in SQL; these cases pin its
classification of added/modified/removed symbols with exact-set
assertions (an unchanged neighbour leaking in fails as loudly as
a missing symbol).
"""

from __future__ import annotations

import pygit2
from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.frames import changed_to_symbols
from rbtr.index.models import ChangeKind, Chunk
from rbtr.index.orchestrator import build_index
from rbtr.index.store import IndexStore

from .asserts import assert_changes
from .cases_diff import DiffScenario
from .conftest import commit_file_set


@fixture
@parametrize_with_cases("scenario", cases=".cases_diff")
def diff_result(
    scenario: DiffScenario,
    diff_repo: pygit2.Repository,
    store: IndexStore,
) -> tuple[DiffScenario, list[tuple[Chunk, ChangeKind]]]:
    base_sha = commit_file_set(diff_repo, scenario.base_files, parents=[])
    build_index(diff_repo.workdir, base_sha, store, repo_id=1)

    if scenario.same_as_base:
        head_sha = base_sha
    else:
        head_commit = commit_file_set(diff_repo, scenario.head_files, parents=[base_sha])
        if scenario.head_as_tree:
            commit_obj = diff_repo.get(head_commit)
            assert isinstance(commit_obj, pygit2.Commit)
            head_sha = str(commit_obj.tree_id)
        else:
            head_sha = head_commit
        build_index(diff_repo.workdir, head_sha, store, repo_id=1, base_sha=base_sha)

    frame = store.diff_symbols(base_sha, head_sha, repo_id=1, file_paths=scenario.file_paths)
    return scenario, changed_to_symbols(frame)


def test_diff_symbols(
    diff_result: tuple[DiffScenario, list[tuple[Chunk, ChangeKind]]],
) -> None:
    scenario, result = diff_result
    assert_changes(
        result,
        added=scenario.expected_added,
        modified=scenario.expected_modified,
        removed=scenario.expected_removed,
    )
