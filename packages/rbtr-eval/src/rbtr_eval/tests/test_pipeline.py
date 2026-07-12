"""Behaviour tests for the measure polars pipeline.

Exercises the composed `_score_outcomes |> _aggregate` chain against
synthetic `SearchBatch` frames built in `cases_pipeline.py`. No daemon,
no embedding model; the purpose is to catch scoring and aggregation
bugs before the expensive indexing run hits them.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from rbtr_eval.measure import _aggregate, _score_outcomes
from rbtr_eval.schemas import SearchBatch

# ── scoring ──────────────────────────────────────────────────────────────────


@parametrize_with_cases("batch", cases=".cases_pipeline", has_tag="scoring")
def test_scoring_assigns_ranks(batch: dy.DataFrame[SearchBatch]) -> None:
    """`_score_outcomes` ranks each target and preserves every input row.

    Covers hit position, line-based disambiguation, the null rank for a
    missing target (with top-1 still populated), and the left-join shape
    (every input row appears exactly once).
    """
    scored = _score_outcomes(batch).sort("query_name")

    # Row preservation: every input row survives exactly once.
    assert scored.height == batch.height

    # Rank per target: found → 1, foo → 2 (line-disambiguated), absent → null.
    by_name = dict(zip(scored["query_name"], scored["rank"], strict=True))
    assert by_name == {"found": 1, "foo": 2, "absent": None}

    # Missing target keeps its top-1 fields populated.
    absent = scored.filter(pl.col("query_name") == "absent").row(0, named=True)
    assert absent["top_file"] == "a.py"
    assert absent["top_name"] == "bar"

    # Line disambiguation: the line-99 hit is top-1 even though the
    # rank-2 match is at line 10.
    foo = scored.filter(pl.col("query_name") == "foo").row(0, named=True)
    assert foo["top_file"] == "q.py"
    assert foo["top_line"] == 99
    assert foo["top_name"] == "foo"


# ── aggregation ──────────────────────────────────────────────────────────────


@parametrize_with_cases("batch", cases=".cases_pipeline", has_tag="metrics")
def test_aggregate_computes_metrics(batch: dy.DataFrame[SearchBatch]) -> None:
    """`score |> aggregate` computes Hit@k / MRR / NDCG and the rollup.

    Slug `r` has ranks 1, 3, null, null: Hit@1 = 1/4, Hit@3 = Hit@10 =
    2/4, MRR = (1 + 1/3 + 0 + 0) / 4, median over {1, 3} = 2.0, and the
    two null ranks count as misses (not_found = 0.5). Two slugs produce
    per-slug rows plus an `__all__` rollup.
    """
    metrics = _aggregate(_score_outcomes(batch))

    per_repo = metrics.filter(
        (pl.col("slug") == "r")
        & (pl.col("language") == "__all__")
        & (pl.col("provenance") == "__all__")
    ).row(0, named=True)
    assert per_repo["n_queries"] == 4
    assert per_repo["hit_at_1"] == 0.25
    assert per_repo["hit_at_3"] == 0.5
    assert per_repo["hit_at_10"] == 0.5
    assert per_repo["mrr"] == (1 + 1 / 3) / 4  # null ranks are zero reciprocal
    assert per_repo["ndcg_at_10"] == pytest.approx(0.375)
    assert per_repo["not_found_pct"] == 0.5
    assert per_repo["median_rank"] == 2.0
    assert per_repo["search_p50_ms"] == 30.0
    assert per_repo["search_p95_ms"] == 40.0

    # Two slugs produce per-slug rows plus the `__all__` rollup.
    assert sorted(metrics["slug"].unique().to_list()) == ["__all__", "r", "r2"]


@parametrize_with_cases("batch", cases=".cases_pipeline", has_tag="kinds")
def test_aggregate_breaks_out_symbol_kind(batch: dy.DataFrame[SearchBatch]) -> None:
    """Metrics carry a per-`symbol_kind` rollup and a `symbol_kind` x
    `query_kind` rollup, so each target kind is measured on its own axis
    and in the cross with request shape."""
    metrics = _aggregate(_score_outcomes(batch))

    per_kind = metrics.filter(
        (pl.col("slug") == "__all__")
        & (pl.col("language") == "__all__")
        & (pl.col("provenance") == "__all__")
        & (pl.col("query_kind") == "__all__")
        & (pl.col("symbol_kind") != "__all__")
    )
    assert set(per_kind["symbol_kind"].to_list()) == {"function", "config_key"}

    cross = metrics.filter(
        (pl.col("slug") == "__all__")
        & (pl.col("language") == "__all__")
        & (pl.col("provenance") == "__all__")
        & (pl.col("symbol_kind") != "__all__")
        & (pl.col("query_kind") != "__all__")
    )
    pairs = set(zip(cross["symbol_kind"].to_list(), cross["query_kind"].to_list(), strict=True))
    assert pairs == {("function", "concept"), ("config_key", "identifier")}
