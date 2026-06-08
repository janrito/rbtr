"""Behavioural tests for `rbtr.index.search.fuse_scores`.

Each scenario in `case_fuse.py` supplies the full input set and
the expected-output property to assert.  The fixture builds a
polars candidates frame from the case's `ChunkSpec` list and
calls `fuse_scores`; each test function asserts one property.
"""

from __future__ import annotations

import polars as pl
import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.config import config
from rbtr.index.classify import QueryKind
from rbtr.index.frames import FusionInputRow
from rbtr.index.models import ScoredChunk
from rbtr.index.search import fuse_scores, materialise_scored

from .cases_fuse import FuseScenario


@fixture
@parametrize_with_cases("scenario")
def fused(scenario: FuseScenario) -> tuple[FuseScenario, list[ScoredChunk]]:
    ids = [s.id for s in scenario.chunks]
    n = len(ids)
    candidates = pl.DataFrame(
        {
            "id": ids,
            "repo_id": [1] * n,
            "blob_sha": ["blob"] * n,
            "file_path": [s.file_path for s in scenario.chunks],
            "kind": [s.kind.value for s in scenario.chunks],
            "name": [s.name for s in scenario.chunks],
            "scope": [""] * n,
            "language": [""] * n,
            "content": [""] * n,
            "content_tokens": [""] * n,
            "name_tokens": [""] * n,
            "line_start": [1] * n,
            "line_end": [1] * n,
            "metadata": [{"module": "", "names": "", "dots": ""}] * n,
            "has_embedding": [False] * n,
            "lexical": [scenario.lexical.get(cid, 0.0) for cid in ids],
            "semantic": [scenario.semantic.get(cid, 0.0) for cid in ids],
            "importance": [1.0] * n,
            "proximity": [1.0] * n,
        },
        schema_overrides={"kind": pl.Categorical, "line_start": pl.Int32, "line_end": pl.Int32},
    )

    validated = FusionInputRow.validate(candidates, cast=True)
    frame = fuse_scores(
        validated,
        query=scenario.query,
        alpha=scenario.alpha
        if scenario.alpha is not None
        else config.search_weights[QueryKind.IDENTIFIER].alpha,
        beta=scenario.beta
        if scenario.beta is not None
        else config.search_weights[QueryKind.IDENTIFIER].beta,
        gamma=scenario.gamma
        if scenario.gamma is not None
        else config.search_weights[QueryKind.IDENTIFIER].gamma,
        top_k=scenario.top_k,
    )
    return scenario, materialise_scored(frame)


def test_count_matches(fused: tuple[FuseScenario, list[ScoredChunk]]) -> None:
    scenario, results = fused
    if scenario.expected_count is None:
        return
    assert len(results) == scenario.expected_count


def test_top_ranked_id_matches(
    fused: tuple[FuseScenario, list[ScoredChunk]],
) -> None:
    scenario, results = fused
    if scenario.expected_top is None:
        return
    assert results, "expected at least one result"
    assert results[0].id == scenario.expected_top


def test_full_ranked_order_matches(
    fused: tuple[FuseScenario, list[ScoredChunk]],
) -> None:
    scenario, results = fused
    if scenario.expected_order is None:
        return
    assert [r.id for r in results] == scenario.expected_order


def test_result_id_set_matches(
    fused: tuple[FuseScenario, list[ScoredChunk]],
) -> None:
    scenario, results = fused
    if scenario.expected_ids is None:
        return
    assert sorted(r.id for r in results) == sorted(scenario.expected_ids)


def test_first_score_approx_matches(
    fused: tuple[FuseScenario, list[ScoredChunk]],
) -> None:
    scenario, results = fused
    if scenario.expected_first_score_approx is None:
        return
    assert results
    assert results[0].score == pytest.approx(scenario.expected_first_score_approx)


def test_every_result_has_populated_breakdown(
    fused: tuple[FuseScenario, list[ScoredChunk]],
) -> None:
    """A universal invariant: every ScoredChunk has positive boosts."""
    _scenario, results = fused
    for r in results:
        assert isinstance(r, ScoredChunk)
        assert r.kind_boost > 0.0
        assert r.file_penalty > 0.0
        assert r.score >= 0.0
        assert r.fusion == r.score
        assert r.reranker == 0.0
