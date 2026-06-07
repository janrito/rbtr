"""Shared rank-based aggregation expressions."""

from __future__ import annotations

import polars as pl


def search_metric_aggs() -> list[pl.Expr]:
    """Rank-based aggregation expressions for search evaluation.

    Depends only on a `rank` column (nullable UInt8, 1-10 or
    null for miss).
    """
    return [
        pl.len().cast(pl.UInt32).alias("n_queries"),
        (pl.col("rank") <= 1).fill_null(False).mean().alias("hit_at_1"),
        (pl.col("rank") <= 3).fill_null(False).mean().alias("hit_at_3"),
        (pl.col("rank") <= 10).fill_null(False).mean().alias("hit_at_10"),
        pl.when(pl.col("rank").is_null())
        .then(0.0)
        .otherwise(1.0 / pl.col("rank").cast(pl.Float64))
        .mean()
        .alias("mrr"),
        pl.when(pl.col("rank").is_null())
        .then(0.0)
        .otherwise(1.0 / (pl.col("rank").cast(pl.Float64) + 1).log(base=2))
        .mean()
        .alias("ndcg_at_10"),
        pl.col("rank").drop_nulls().cast(pl.Float64).median().alias("median_rank"),
        pl.col("rank").is_null().mean().alias("not_found_pct"),
    ]
