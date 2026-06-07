"""Behavioural tests for `rbtr.index.reranker.Reranker.rerank`.

Each scenario in `cases_reranker.py` supplies pre-fused candidates,
stub model scores, and expected output properties.  The fixture
builds a `FusedRow` frame, injects a stub model via `model_loader`,
and calls `rerank()`.
"""

from __future__ import annotations

from typing import Any

import dataframely as dy
import polars as pl
import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.frames import FusedRow
from rbtr.index.reranker import Reranker

from .cases_reranker import RerankScenario

# ── Stub model ───────────────────────────────────────────────────────


class _StubRerankerModel:
    """Minimal stand-in for `llama_cpp.Llama` with rank pooling.

    Looks up the chunk content in the prompt to find the matching
    id in the scenario's `reranker_scores` dict.
    """

    def __init__(
        self,
        scores: dict[str, float],
        content_to_id: dict[str, str],
        *,
        should_raise: bool = False,
    ) -> None:
        self._scores = scores
        self._content_to_id = content_to_id
        self._should_raise = should_raise

    def embed(self, prompt: str) -> list[float]:
        if self._should_raise:
            msg = "stub reranker failure"
            raise RuntimeError(msg)
        for content, cid in self._content_to_id.items():
            if content in prompt:
                return [self._scores.get(cid, 0.0)]
        return [0.0]

    def close(self) -> None:
        pass


# ── Helpers ──────────────────────────────────────────────────────────


def _build_fused_frame(scenario: RerankScenario) -> dy.DataFrame[FusedRow]:
    """Build a `dy.DataFrame[FusedRow]` from scenario candidates."""
    if not scenario.candidates:
        return FusedRow.create_empty()

    ids = list(scenario.candidates.keys())
    n = len(ids)
    fusions = [scenario.candidates[cid][0] for cid in ids]
    contents = [scenario.candidates[cid][1] for cid in ids]

    frame = pl.DataFrame(
        {
            "id": ids,
            "repo_id": [1] * n,
            "blob_sha": ["blob"] * n,
            "file_path": ["src/lib.py"] * n,
            "kind": ["function"] * n,
            "name": ["fn"] * n,
            "scope": [""] * n,
            "language": [""] * n,
            "content": contents,
            "line_start": [1] * n,
            "line_end": [1] * n,
            "metadata": [{"module": "", "names": "", "dots": ""}] * n,
            "lexical": [0.0] * n,
            "semantic": [0.0] * n,
            "importance": [1.0] * n,
            "proximity": [1.0] * n,
            "embedding": [[]] * n,
            "score": fusions,
            "name_match": [0.0] * n,
            "kind_boost": [1.0] * n,
            "file_penalty": [1.0] * n,
            "fusion": fusions,
            "reranker": [0.0] * n,
        },
        schema_overrides={
            "kind": pl.Categorical,
            "line_start": pl.Int32,
            "line_end": pl.Int32,
        },
    )
    return FusedRow.validate(frame, cast=True)


def _make_reranker(scenario: RerankScenario) -> Reranker:
    """Build a `Reranker` with a stub model from the scenario."""
    content_to_id = {scenario.candidates[cid][1]: cid for cid in scenario.candidates}
    stub = _StubRerankerModel(
        scores=scenario.reranker_scores,
        content_to_id=content_to_id,
        should_raise=scenario.model_raises,
    )
    return Reranker(model_loader=lambda: stub)  # type: ignore[arg-type,return-value]  # stub satisfies Llama.embed interface


# ── Fixture ──────────────────────────────────────────────────────────


@fixture
@parametrize_with_cases("scenario")
def reranked(scenario: RerankScenario) -> tuple[RerankScenario, list[dict[str, Any]]]:
    """Run `rerank()` and return (scenario, result rows as dicts)."""
    frame = _build_fused_frame(scenario)
    reranker = _make_reranker(scenario)
    try:
        result = reranker.rerank(
            scenario.query,
            frame,
            top_k=scenario.top_k,
            blend_weight=scenario.blend_weight,
        )
    finally:
        reranker.close()
    return scenario, result.to_dicts()


# ── Tests ────────────────────────────────────────────────────────────


def test_count(reranked: tuple[RerankScenario, list[dict[str, Any]]]) -> None:
    scenario, rows = reranked
    if scenario.expected_count is None:
        return
    assert len(rows) == scenario.expected_count


def test_top_ranked_id(reranked: tuple[RerankScenario, list[dict[str, Any]]]) -> None:
    scenario, rows = reranked
    if scenario.expected_top is None:
        return
    assert rows, "expected at least one result"
    assert rows[0]["id"] == scenario.expected_top


def test_ranked_order(reranked: tuple[RerankScenario, list[dict[str, Any]]]) -> None:
    scenario, rows = reranked
    if scenario.expected_order is None:
        return
    assert [r["id"] for r in rows] == scenario.expected_order


def test_score_breakdown(reranked: tuple[RerankScenario, list[dict[str, Any]]]) -> None:
    """Universal invariant: fusion unchanged, reranker ≥ 0, blend correct."""
    scenario, rows = reranked
    w = scenario.blend_weight
    for r in rows:
        # fusion column must not be modified by rerank
        assert r["fusion"] == scenario.candidates[r["id"]][0]
        assert r["reranker"] >= 0.0

        if scenario.model_raises:
            assert r["reranker"] == 0.0
            # score is fusion (fallback)
            assert r["score"] == pytest.approx(r["fusion"])
        else:
            assert r["score"] == pytest.approx(w * r["fusion"] + (1 - w) * r["reranker"])
