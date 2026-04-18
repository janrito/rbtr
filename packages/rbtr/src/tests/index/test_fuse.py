"""Behavioural tests for ``rbtr.index.search.fuse_scores``.

Each scenario in ``case_fuse.py`` supplies the full input set and
the expected-output property to assert.  The fixture turns the
case's ``ChunkSpec`` list into real ``Chunk`` objects and calls
``fuse_scores`` with the case's weights; each test function
asserts one property.
"""

from __future__ import annotations

import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.models import Chunk
from rbtr.index.search import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    ScoredResult,
    fuse_scores,
)
from tests.index.case_fuse import FuseCase


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_fuse")
def fused(scenario: FuseCase) -> tuple[FuseCase, list[ScoredResult]]:
    chunks = {
        spec.id: Chunk(
            id=spec.id,
            blob_sha="blob",
            file_path=spec.file_path,
            kind=spec.kind,
            name=spec.name,
            content="",
            line_start=1,
            line_end=1,
        )
        for spec in scenario.chunks
    }
    results = fuse_scores(
        chunks,
        lexical_scores=scenario.lexical,
        semantic_scores=scenario.semantic,
        name_scores=scenario.name,
        alpha=DEFAULT_ALPHA if scenario.alpha is None else scenario.alpha,
        beta=DEFAULT_BETA if scenario.beta is None else scenario.beta,
        gamma=DEFAULT_GAMMA if scenario.gamma is None else scenario.gamma,
        top_k=scenario.top_k,
    )
    return scenario, results


def test_count_matches(fused: tuple[FuseCase, list[ScoredResult]]) -> None:
    scenario, results = fused
    if scenario.expected_count is None:
        return
    assert len(results) == scenario.expected_count


def test_top_ranked_id_matches(
    fused: tuple[FuseCase, list[ScoredResult]],
) -> None:
    scenario, results = fused
    if scenario.expected_top is None:
        return
    assert results, "expected at least one result"
    assert results[0].chunk.id == scenario.expected_top


def test_full_ranked_order_matches(
    fused: tuple[FuseCase, list[ScoredResult]],
) -> None:
    scenario, results = fused
    if scenario.expected_order is None:
        return
    assert [r.chunk.id for r in results] == scenario.expected_order


def test_result_id_set_matches(
    fused: tuple[FuseCase, list[ScoredResult]],
) -> None:
    scenario, results = fused
    if scenario.expected_ids is None:
        return
    assert sorted(r.chunk.id for r in results) == sorted(scenario.expected_ids)


def test_first_score_approx_matches(
    fused: tuple[FuseCase, list[ScoredResult]],
) -> None:
    scenario, results = fused
    if scenario.expected_first_score_approx is None:
        return
    assert results
    assert results[0].score == pytest.approx(scenario.expected_first_score_approx)


def test_every_result_has_populated_breakdown(
    fused: tuple[FuseCase, list[ScoredResult]],
) -> None:
    """A universal invariant: every ScoredResult has positive boosts."""
    _scenario, results = fused
    for r in results:
        assert isinstance(r, ScoredResult)
        assert r.kind_boost > 0.0
        assert r.file_penalty > 0.0
        assert r.score >= 0.0
