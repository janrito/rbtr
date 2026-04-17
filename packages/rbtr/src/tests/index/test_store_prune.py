"""Behavioural tests for ``prune_orphans`` and ``count_orphan_chunks``."""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.store import IndexStore
from tests.index.case_store_prune import PruneScenario


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_store_prune")
def seeded(scenario: PruneScenario) -> tuple[IndexStore, PruneScenario]:
    store = IndexStore()
    store.register_repo("/default")
    if scenario.chunks:
        store.insert_chunks(list(scenario.chunks))
    for commit, path, blob in scenario.snapshots:
        store.insert_snapshot(commit, path, blob)
    by_commit: dict[str, list] = {}
    for commit, edge in scenario.edges:
        by_commit.setdefault(commit, []).append(edge)
    for commit, edges in by_commit.items():
        store.insert_edges(list(edges), commit)
    return store, scenario


def test_count_orphan_chunks_before_prune(
    seeded: tuple[IndexStore, PruneScenario],
) -> None:
    store, scenario = seeded
    assert store.count_orphan_chunks() == scenario.expected_orphan_count_before


def test_prune_orphans_returns_expected_counts(
    seeded: tuple[IndexStore, PruneScenario],
) -> None:
    store, scenario = seeded
    chunks_del, edges_del = store.prune_orphans()
    assert chunks_del == scenario.expected_prune_chunks
    assert edges_del == scenario.expected_prune_edges


def test_count_orphan_chunks_after_prune_is_zero(
    seeded: tuple[IndexStore, PruneScenario],
) -> None:
    store, scenario = seeded
    store.prune_orphans()
    assert store.count_orphan_chunks() == scenario.expected_orphan_count_after


def test_surviving_chunks_match_scenario(
    seeded: tuple[IndexStore, PruneScenario],
) -> None:
    store, scenario = seeded
    store.prune_orphans()
    # Query across every commit in the scenario to collect survivors.
    commits = {sha for sha, _p, _b in scenario.snapshots}
    surviving_ids: set[str] = set()
    for commit in commits:
        for chunk in store.get_chunks(commit):
            surviving_ids.add(chunk.id)
    assert surviving_ids == set(scenario.expected_surviving_chunk_ids)
