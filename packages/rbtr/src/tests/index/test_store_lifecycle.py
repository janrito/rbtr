"""Behavioural tests for ``indexed_commits`` tracking."""

from __future__ import annotations

import time

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.store import IndexStore
from tests.index.case_store_lifecycle import LifecycleScenario


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_store_lifecycle")
def seeded(scenario: LifecycleScenario) -> tuple[IndexStore, LifecycleScenario]:
    store = IndexStore()
    for i, path in enumerate(scenario.repo_paths):
        repo_id = store.register_repo(path)
        assert repo_id == i + 1, "cases rely on sequential repo_ids"
    for i, (repo_id, sha) in enumerate(scenario.marks):
        store.mark_indexed(repo_id, sha)
        if i in scenario.sleep_after:
            time.sleep(0.01)
    return store, scenario


def test_has_indexed_matches_scenario(
    seeded: tuple[IndexStore, LifecycleScenario],
) -> None:
    store, scenario = seeded
    for (repo_id, sha), expected in scenario.has_indexed.items():
        assert store.has_indexed(repo_id, sha) is expected, (
            f"repo_id={repo_id}, sha={sha!r}"
        )


def test_list_indexed_commits_order_matches_scenario(
    seeded: tuple[IndexStore, LifecycleScenario],
) -> None:
    store, scenario = seeded
    for repo_id, expected_order in scenario.list_order.items():
        actual = [sha for sha, _ in store.list_indexed_commits(repo_id)]
        assert actual == expected_order, f"repo_id={repo_id}"
