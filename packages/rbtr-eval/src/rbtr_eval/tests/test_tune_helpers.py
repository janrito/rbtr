"""Tests for tune helper functions: _rescore_and_rank, _simplex_from_unit_square."""

from __future__ import annotations

import dataframely as dy
import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from rbtr_eval.queries import PROVENANCE_TO_KIND
from rbtr_eval.schemas import QueryMeta, QueryRow, ScoredCandidate, TuneReport
from rbtr_eval.tune import (
    _hmean_mrr,
    _mrr_per_provenance,
    _rescore_and_rank,
    _simplex_from_unit_square,
    _toml_snippet,
    _with_query_kind,
)

# ── _rescore_and_rank ────────────────────────────────────────────────────────


@parametrize_with_cases(
    "candidates, queries, weights, expected_ranks, expected_mrr",
    has_tag="rescore",
)
def test_rescore_and_rank(
    candidates: dy.DataFrame[ScoredCandidate],
    queries: dy.DataFrame[QueryMeta],
    weights: tuple[float, float, float],
    expected_ranks: list[int | None],
    expected_mrr: float,
) -> None:
    """Polars re-scoring produces expected ranks."""
    result = _rescore_and_rank(candidates, queries, weights)
    assert result["rank"].to_list() == expected_ranks


@parametrize_with_cases(
    "candidates, queries, weights, expected_ranks, expected_mrr",
    has_tag="rescore",
)
def test_rescore_objective(
    candidates: dy.DataFrame[ScoredCandidate],
    queries: dy.DataFrame[QueryMeta],
    weights: tuple[float, float, float],
    expected_ranks: list[int | None],
    expected_mrr: float,
) -> None:
    """Full objective chain: rescore -> per-provenance MRR -> harmonic mean."""
    ranks = _rescore_and_rank(candidates, queries, weights)
    mrr = ranks.pipe(_mrr_per_provenance).pipe(_hmean_mrr)
    assert mrr == pytest.approx(expected_mrr, abs=1e-6)


# ── _with_query_kind ───────────────────────────────────────────────────


@parametrize_with_cases("queries, expected_kinds", has_tag="query_kind")
def test_with_query_kind(
    queries: dy.DataFrame[QueryRow],
    expected_kinds: list[str],
) -> None:
    """Provenance values map to the correct query kinds."""
    result = _with_query_kind(queries)
    assert result["query_kind"].to_list() == expected_kinds


def test_provenance_mapping_covers_all_eval_provenances() -> None:
    """Every eval provenance has a mapping entry."""
    eval_provenances = {"name", "body", "docstring", "concept"}
    assert eval_provenances == set(PROVENANCE_TO_KIND.keys())


# ── _simplex_from_unit_square ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("u", "v", "expected"),
    [
        (0.0, 0.0, (0.0, 0.0, 1.0)),
        (1.0, 0.0, (1.0, 0.0, 0.0)),
        (0.0, 1.0, (0.0, 1.0, 0.0)),
        (0.5, 0.5, (0.5, 0.25, 0.25)),
    ],
)
def test_simplex_maps_correctly(u: float, v: float, expected: tuple[float, float, float]) -> None:
    """Unit-square points map to expected simplex coordinates."""
    result = _simplex_from_unit_square(u, v)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("u", "v"),
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.3, 0.7), (0.9, 0.1)],
)
def test_simplex_sums_to_one(u: float, v: float) -> None:
    """Every mapped triple sums to 1."""
    a, b, g = _simplex_from_unit_square(u, v)
    assert a + b + g == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("u", "v"),
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.3, 0.7), (0.9, 0.1)],
)
def test_simplex_components_non_negative(u: float, v: float) -> None:
    """All components are in [0, 1]."""
    a, b, g = _simplex_from_unit_square(u, v)
    assert 0.0 <= a <= 1.0
    assert 0.0 <= b <= 1.0
    assert 0.0 <= g <= 1.0


# ── _toml_snippet ────────────────────────────────────────────────────────────


def test_toml_snippet_per_kind() -> None:
    """Per-kind report emits [search_weights.<kind>] sections."""
    report = pl.DataFrame(
        {
            "kind": ["concept", "identifier"],
            "best_alpha": [0.5, 0.05],
            "best_beta": [0.3, 0.2],
            "best_gamma": [0.2, 0.75],
            "score_best": [0.4, 0.5],
            "current_alpha": [0.1, 0.1],
            "current_beta": [0.3, 0.3],
            "current_gamma": [0.6, 0.6],
            "score_current": [0.3, 0.4],
            "delta": [0.1, 0.1],
            "metric": ["MRR", "MRR"],
            "n_trials": [10, 10],
            "n_queries": [50, 50],
            "elapsed_seconds": [1.0, 1.0],
        }
    ).pipe(TuneReport.validate, cast=True)
    snippet = _toml_snippet(report)
    assert "[search_weights.concept]" in snippet
    assert "[search_weights.identifier]" in snippet
    assert "alpha = 0.5" in snippet
    assert "alpha = 0.05" in snippet
