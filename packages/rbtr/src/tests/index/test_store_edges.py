"""Behavioural tests for edge storage and retrieval.

One shared fixture seeds the store from an ``EdgeScenario``.
Each test function asserts one dimension of the scenario's
expectations (plain read, by-source filter, by-kind filter).
Scenarios that leave a dimension unspecified are silently
skipped for the test covering that dimension.
"""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.store import IndexStore
from tests.index.case_store_edges import EdgeScenario


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_store_edges")
def seeded(scenario: EdgeScenario) -> tuple[IndexStore, EdgeScenario]:
    """Build an ``IndexStore`` populated per *scenario*."""
    store = IndexStore()
    for i, path in enumerate(scenario.repo_paths):
        repo_id = store.register_repo(path)
        assert repo_id == i + 1, "cases rely on sequential repo_ids"
        data = scenario.per_repo[i] if i < len(scenario.per_repo) else None
        if data is None:
            continue
        for commit, edges in data.per_commit.items():
            if edges:
                store.insert_edges(list(edges), commit, repo_id=repo_id)
    return store, scenario


def test_get_edges_returns_expected(
    seeded: tuple[IndexStore, EdgeScenario],
) -> None:
    store, scenario = seeded
    for (repo_id, commit), expected in scenario.expected_edges.items():
        actual = store.get_edges(commit, repo_id=repo_id)
        assert actual == expected, f"repo_id={repo_id}, commit={commit!r}"


def test_get_edges_filter_by_source(
    seeded: tuple[IndexStore, EdgeScenario],
) -> None:
    store, scenario = seeded
    for (repo_id, commit, source), expected in scenario.expected_by_source.items():
        actual = store.get_edges(commit, source_id=source, repo_id=repo_id)
        assert actual == expected, f"repo_id={repo_id}, commit={commit!r}, source_id={source!r}"


def test_get_edges_filter_by_kind(
    seeded: tuple[IndexStore, EdgeScenario],
) -> None:
    store, scenario = seeded
    for (repo_id, commit, kind), expected in scenario.expected_by_kind.items():
        actual = store.get_edges(commit, kind=kind, repo_id=repo_id)
        assert actual == expected, f"repo_id={repo_id}, commit={commit!r}, kind={kind!r}"
