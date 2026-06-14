"""Behaviour tests and invariant guards for the tune objective.

The objective `_rescore_and_rank |> _mrr_per_provenance |> _hmean_mrr`
is what Bayesian optimisation drives; the tests prove a better weight
triple scores higher than a worse one, and guard the invariants that
keep the search well-formed (a valid probability simplex, a complete
provenance→kind mapping, and a paste-able TOML report).
"""

from __future__ import annotations

import tomllib

import dataframely as dy
import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from rbtr_eval.queries import PROVENANCE_TO_KIND
from rbtr_eval.schemas import QueryMeta, ScoredCandidate, TuneReport
from rbtr_eval.tune import (
    _hmean_mrr,
    _mrr_per_provenance,
    _rescore_and_rank,
    _simplex_from_unit_square,
    _toml_snippet,
)

# ── Objective ────────────────────────────────────────────────────────────────


@parametrize_with_cases(
    "candidates, queries, weights, expected_ranks, expected_mrr",
    has_tag="rescore",
)
def test_objective_scores_weight_triples(
    candidates: dy.DataFrame[ScoredCandidate],
    queries: dy.DataFrame[QueryMeta],
    weights: tuple[float, float, float],
    expected_ranks: list[int | None],
    expected_mrr: float,
) -> None:
    """Re-scoring ranks the target and the objective MRR reflects it.

    Asserts both the intermediate ranks (localises a re-ranking bug)
    and the end MRR (the value Optuna optimises). The cases include the
    pair where semantic-heavy vs lexical-heavy weights flip the target
    between rank 1 (MRR 1.0) and rank 2 (MRR 0.5) — the behaviour that
    makes tuning worthwhile — plus the empty-candidates and
    outside-top-10 guards (null rank, floored MRR, no crash).
    """
    ranks = _rescore_and_rank(candidates, queries, weights)
    assert ranks["rank"].to_list() == expected_ranks

    mrr = ranks.pipe(_mrr_per_provenance).pipe(_hmean_mrr)
    assert mrr == pytest.approx(expected_mrr, abs=1e-6)


# ── Invariant guards ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("u", "v", "expected"),
    [
        (0.0, 0.0, (0.0, 0.0, 1.0)),
        (1.0, 0.0, (1.0, 0.0, 0.0)),
        (0.0, 1.0, (0.0, 1.0, 0.0)),
        (0.5, 0.5, (0.5, 0.25, 0.25)),
    ],
)
def test_simplex_maps_unit_square_to_valid_triple(
    u: float, v: float, expected: tuple[float, float, float]
) -> None:
    """Unit-square points map to the specific simplex point, and the
    result is always a valid probability triple (sums to 1, in `[0, 1]`)
    — guarding against an invalid weight triple reaching search.
    """
    result = _simplex_from_unit_square(u, v)
    assert result == pytest.approx(expected)
    assert sum(result) == pytest.approx(1.0)
    assert all(0.0 <= c <= 1.0 for c in result)


def test_every_provenance_has_a_query_kind() -> None:
    """Every eval provenance maps to a query kind.

    Guards against adding a provenance without a mapping entry, which
    would silently misclassify or drop those queries during tuning.
    """
    assert set(PROVENANCE_TO_KIND.keys()) == {"name", "body", "docstring", "concept"}


def test_toml_snippet_is_valid_toml() -> None:
    """The report snippet parses as TOML and round-trips the weights.

    Guards against emitting config the operator cannot paste into
    `rbtr`'s settings.
    """
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

    parsed = tomllib.loads(_toml_snippet(report))

    weights = parsed["search_weights"]
    assert weights["concept"] == {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
    assert weights["identifier"] == {"alpha": 0.05, "beta": 0.2, "gamma": 0.75}
