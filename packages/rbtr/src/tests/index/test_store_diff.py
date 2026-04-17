"""Behavioural tests for ``IndexStore.diff_chunks``."""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.store import IndexStore
from tests.index.case_store_diff import DiffScenario


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_store_diff")
def seeded(scenario: DiffScenario) -> tuple[IndexStore, DiffScenario]:
    store = IndexStore()
    store.register_repo("/default")
    if scenario.chunks:
        store.insert_chunks(list(scenario.chunks))
    for commit, path, blob in scenario.snapshots:
        store.insert_snapshot(commit, path, blob)
    return store, scenario


def test_diff_chunks_returns_expected_ids(
    seeded: tuple[IndexStore, DiffScenario],
) -> None:
    store, scenario = seeded
    added, removed, modified = store.diff_chunks(scenario.base, scenario.head)
    assert sorted(c.id for c in added) == sorted(scenario.expected_added)
    assert sorted(c.id for c in removed) == sorted(scenario.expected_removed)
    assert sorted(c.id for c in modified) == sorted(scenario.expected_modified)
