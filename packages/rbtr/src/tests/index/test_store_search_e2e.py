"""End-to-end tests for ``IndexStore.search()``.

These exercise the full fusion pipeline through a real FTS index,
as opposed to ``test_fuse.py`` which tests the pure function.  Cases
live in ``case_store_search_e2e.py``.
"""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.models import Chunk
from rbtr.index.search import ScoredResult
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code
from tests.index.case_store_search_e2e import ChunkSpec, StoreSearchCase


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_store_search_e2e")
def searched(
    scenario: StoreSearchCase,
) -> tuple[StoreSearchCase, list[ScoredResult]]:
    store = IndexStore()
    chunks = [
        Chunk(
            id=spec.id,
            blob_sha=f"blob_{spec.id}",
            file_path=spec.file_path,
            kind=spec.kind,
            name=spec.name,
            content=spec.content,
            content_tokens=tokenise_code(spec.content),
            name_tokens=tokenise_code(spec.name),
            line_start=1,
            line_end=1,
        )
        for spec in scenario.chunks
    ]
    store.insert_chunks(chunks)
    for chunk in chunks:
        store.insert_snapshot("head", chunk.file_path, chunk.blob_sha)
    if scenario.edges:
        store.insert_edges(list(scenario.edges), "head")
    changed = set(scenario.changed_files) if scenario.changed_files else None
    results = store.search(
        "head",
        scenario.query,
        top_k=scenario.top_k,
        changed_files=changed,
    )
    return scenario, results


def test_result_count_at_least(
    searched: tuple[StoreSearchCase, list[ScoredResult]],
) -> None:
    scenario, results = searched
    if scenario.expected_count_at_least is None:
        return
    if scenario.expected_count_at_least == 0:
        assert results == []
    else:
        assert len(results) >= scenario.expected_count_at_least


def test_top_ranked_id_matches(
    searched: tuple[StoreSearchCase, list[ScoredResult]],
) -> None:
    scenario, results = searched
    if scenario.expected_top is None:
        return
    assert results, "expected at least one result"
    assert results[0].chunk.id == scenario.expected_top


def test_first_of_kind_matches(
    searched: tuple[StoreSearchCase, list[ScoredResult]],
) -> None:
    scenario, results = searched
    if scenario.expected_first_of_kind is None:
        return
    expected_id, expected_kind = scenario.expected_first_of_kind
    of_kind = [r for r in results if r.chunk.kind == expected_kind]
    assert of_kind, f"no {expected_kind} results"
    assert of_kind[0].chunk.id == expected_id


def test_first_importance_above_second(
    searched: tuple[StoreSearchCase, list[ScoredResult]],
) -> None:
    scenario, results = searched
    if not scenario.expected_first_importance_above_second:
        return
    of_kind = [
        r
        for r in results
        if scenario.expected_first_of_kind is not None
        and r.chunk.kind == scenario.expected_first_of_kind[1]
    ]
    assert len(of_kind) >= 2
    assert of_kind[0].importance > of_kind[1].importance


def test_first_proximity_above_second(
    searched: tuple[StoreSearchCase, list[ScoredResult]],
) -> None:
    scenario, results = searched
    if not scenario.expected_first_proximity_above_second:
        return
    assert len(results) >= 2
    assert results[0].proximity > results[1].proximity


def test_breakdown_populated(
    searched: tuple[StoreSearchCase, list[ScoredResult]],
) -> None:
    scenario, results = searched
    if scenario.check_breakdown_for_id is None:
        return
    target = next(
        (r for r in results if r.chunk.id == scenario.check_breakdown_for_id),
        None,
    )
    assert target is not None
    assert target.score >= 0.0
    assert target.kind_boost > 0.0
    assert target.file_penalty > 0.0
    assert target.importance >= 1.0
    assert target.proximity >= 1.0
