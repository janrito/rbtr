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

from rbtr_eval.queries import with_query_kind
from rbtr_eval.schemas import QueryMeta, QueryRow, ScoredCandidate, TuneReport
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


@pytest.fixture
def query_kind_rows() -> dy.DataFrame[QueryRow]:
    """Queries whose text shape disagrees with their provenance.

    The same concept-shaped text appears under two provenances, and
    an identifier- and a code-shaped query carry provenances a
    provenance-based mapping would have labelled differently.
    """
    base = {"slug": "r", "scope": "", "symbol_kind": "function", "language": "python"}
    rows = [
        {
            **base,
            "file_path": "a.py",
            "name": "a",
            "line_start": 1,
            "provenance": "body",
            "text": "how does the config loader resolve paths",
        },
        {
            **base,
            "file_path": "a.py",
            "name": "a",
            "line_start": 1,
            "provenance": "docstring",
            "text": "how does the config loader resolve paths",
        },
        {
            **base,
            "file_path": "b.py",
            "name": "b",
            "line_start": 1,
            "provenance": "docstring",
            "text": "fuse_scores",
        },
        {
            **base,
            "file_path": "c.py",
            "name": "c",
            "line_start": 1,
            "provenance": "name",
            "text": "def fuse_scores(candidates, query, *, alpha):",
        },
    ]
    return pl.DataFrame(rows).pipe(QueryRow.validate, cast=True)


def test_query_kind_follows_text_not_provenance(
    query_kind_rows: dy.DataFrame[QueryRow],
) -> None:
    """A query's kind is a function of its request text, not its provenance.

    Tuning must partition queries the way production routes them — by
    `classify_query(text)` — so the same text gets the same kind
    regardless of how the query was generated, and the kind tracks the
    text's shape (concept / identifier / code), not the provenance.
    """
    tagged = with_query_kind(query_kind_rows)
    by_text = dict(zip(tagged["text"], tagged["query_kind"], strict=True))

    # Provenance-independence: same text under body and docstring agrees.
    concept_rows = tagged.filter(pl.col("text") == "how does the config loader resolve paths")
    assert concept_rows["query_kind"].n_unique() == 1

    # Kind tracks the text's shape, not the provenance.
    assert by_text["how does the config loader resolve paths"] == "concept"
    assert by_text["fuse_scores"] == "identifier"
    assert by_text["def fuse_scores(candidates, query, *, alpha):"] == "code"


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
