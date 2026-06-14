"""Integration tests for the measure polars pipeline.

Exercises the composed `_score_outcomes |> _aggregate` chain against
synthetic `SearchBatch` frames. No daemon, no embedding model; the
purpose is to catch scoring and aggregation bugs before the expensive
indexing run hits them.

Two behaviours, two fixtures:

* `scoring_batch` proves ranking detail — hit position, line-based
  disambiguation, missing targets, and row preservation.
* `metrics_batch` proves the aggregated metrics over a known-good rank
  distribution (ranks 1, 3, null, null gives verifiable Hit@k / MRR /
  NDCG), plus the multi-slug rollup.

The fixtures assemble their rows with the `hit` / `outcome_row`
builders from conftest so the bodies stay readable.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl
import pytest

from rbtr_eval.measure import _aggregate, _score_outcomes
from rbtr_eval.schemas import SearchBatch

from .conftest import hit, outcome_row


@pytest.fixture
def scoring_batch() -> dy.DataFrame[SearchBatch]:
    """Three queries exercising every ranking behaviour.

    * `foo` has `query_line_start=10`; only the hit at line 10 matches,
      so the line-99 hit (same file/scope/name) does not, and the
      target lands at rank 2 — covering line-based disambiguation
      (e.g. getters / setters of the same name at different lines).
    * `absent` is missing from its hits — null rank, top-1 still set.
    * `found` sits first in its hits — rank 1.
    """
    return pl.DataFrame(
        [
            outcome_row(
                slug="r",
                target="foo",
                latency_ms=1.0,
                query_line_start=10,
                hits=[hit("q.py", "", "foo", 99), hit("q.py", "", "foo", 10)],
            ),
            outcome_row(
                slug="r",
                target="absent",
                latency_ms=1.0,
                hits=[hit("a.py", "", "bar"), hit("b.py", "", "baz")],
            ),
            outcome_row(
                slug="r",
                target="found",
                latency_ms=1.0,
                hits=[hit("q.py", "", "found")],
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)


@pytest.fixture
def metrics_batch() -> dy.DataFrame[SearchBatch]:
    """Five queries across two slugs with a known-good rank distribution.

    Slug `r` has ranks 1, 3, null, null and latencies 10/20/30/40;
    slug `r2` contributes one rank-1 query so the `__all__` rollup spans
    more than one repo.
    """
    return pl.DataFrame(
        [
            outcome_row(slug="r", target="a", latency_ms=10.0, hits=[hit("q.py", "", "a")]),
            outcome_row(
                slug="r",
                target="b",
                latency_ms=20.0,
                hits=[hit("x.py", "", "x"), hit("y.py", "", "y"), hit("q.py", "", "b")],
            ),
            outcome_row(slug="r", target="c", latency_ms=30.0, hits=[hit("x.py", "", "x")]),
            outcome_row(slug="r", target="d", latency_ms=40.0, hits=[hit("x.py", "", "x")]),
            outcome_row(slug="r2", target="e", latency_ms=5.0, hits=[hit("q.py", "", "e")]),
        ]
    ).pipe(SearchBatch.validate, cast=True)


# ── scoring ──────────────────────────────────────────────────────────────────


def test_scoring_assigns_ranks(scoring_batch: dy.DataFrame[SearchBatch]) -> None:
    """`_score_outcomes` ranks each target and preserves every input row.

    Covers hit position, line-based disambiguation, the null rank for a
    missing target (with top-1 still populated), and the left-join shape
    (every input row appears exactly once).
    """
    scored = _score_outcomes(scoring_batch).sort("query_name")

    # Row preservation: every input row survives exactly once.
    assert scored.height == scoring_batch.height

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


def test_aggregate_computes_metrics(metrics_batch: dy.DataFrame[SearchBatch]) -> None:
    """`score |> aggregate` computes Hit@k / MRR / NDCG and the rollup.

    Slug `r` has ranks 1, 3, null, null: Hit@1 = 1/4, Hit@3 = Hit@10 =
    2/4, MRR = (1 + 1/3 + 0 + 0) / 4, median over {1, 3} = 2.0, and the
    two null ranks count as misses (not_found = 0.5). Two slugs produce
    per-slug rows plus an `__all__` rollup.
    """
    metrics = _aggregate(_score_outcomes(metrics_batch))

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
