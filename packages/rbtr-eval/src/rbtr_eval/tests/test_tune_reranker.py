"""Behaviour test for the reranker's latency-aware selection.

The report is a manual decision aid, and the pick it offers comes from
`_tolerance_sweep`: as the MRR tolerance loosens it walks to smaller —
and therefore faster — pools. Latency is monotonic in pool.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl
import pytest

from rbtr_eval.tune_reranker import GridMetrics, RerankerBreakdown, _tolerance_sweep


@pytest.fixture
def reranker_configs() -> dy.DataFrame[GridMetrics]:
    """A (pool, blend) grid where a bigger pool is marginally better.

    Best MRR (0.90) is at pool 80; pool 50 is within 1% and pool 20
    within 3%, so the pick steps down to smaller pools as tolerance
    loosens. p50 latency rises with pool. At pool 20 two blends tie on
    MRR; the lower blend weight is picked.
    """
    mrr = [0.90, 0.895, 0.88, 0.88, 0.70, 0.69, 0.68]
    return pl.DataFrame(
        {
            "pool": [80, 50, 20, 20, 80, 50, 20],
            "blend_weight": [0.5, 0.5, 0.75, 0.5, 0.0, 0.0, 0.0],
            "n_queries": [100] * 7,
            "hit_at_1": mrr,
            "hit_at_3": mrr,
            "hit_at_10": mrr,
            "mrr": mrr,
            "ndcg_at_10": mrr,
            "median_rank": [1.0] * 7,
            "not_found_pct": [0.1] * 7,
            "search_p50_ms": [7000, 4800, 2100, 2100, 7000, 4800, 2100],
            "search_p95_ms": [9000, 6000, 3000, 3000, 9000, 6000, 3000],
        }
    ).pipe(GridMetrics.validate, cast=True)


@pytest.fixture
def one_kind(reranker_configs: dy.DataFrame[GridMetrics]) -> dy.DataFrame[RerankerBreakdown]:
    """The same grid as the breakdown of one query kind, `concept`."""
    return (
        reranker_configs.drop("search_p95_ms")
        .with_columns(pl.lit("query_kind").alias("dimension"), pl.lit("concept").alias("value"))
        .pipe(RerankerBreakdown.validate, cast=True)
    )


@pytest.mark.parametrize(
    ("tolerance_pct", "expected_pool"),
    [(0.5, 80), (1.0, 50), (3.0, 20)],
)
def test_the_sweep_walks_to_faster_pools(
    reranker_configs: dy.DataFrame[GridMetrics],
    one_kind: dy.DataFrame[RerankerBreakdown],
    tolerance_pct: float,
    expected_pool: int,
) -> None:
    """Looser MRR tolerance selects a smaller, faster pool, at its best blend."""
    sweep = _tolerance_sweep(reranker_configs, one_kind)
    [pick] = sweep.filter(
        pl.col("scope") == "all", (pl.col("tolerance_pct") - tolerance_pct).abs() < 1e-9
    ).rows(named=True)
    assert pick["pool"] == expected_pool
    assert pick["blend_weight"] == 0.5
