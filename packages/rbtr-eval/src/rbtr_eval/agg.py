"""Shared rank-based aggregation expressions and the figures they compute."""

from __future__ import annotations

from enum import StrEnum

import dataframely as dy
import polars as pl


class Dimension(StrEnum):
    """A column a breakdown groups queries by; each value is the column's name."""

    SLUG = "slug"
    LANGUAGE = "language"
    PROVENANCE = "provenance"
    SYMBOL_KIND = "symbol_kind"
    QUERY_KIND = "query_kind"


class SearchMetrics(dy.Schema):
    """The figures `search_metric_aggs` computes over a set of searches.

    `median_rank` is null when every search missed.
    """

    n_queries = dy.UInt32(min=0)
    hit_at_1 = dy.Float64(min=0.0, max=1.0)
    hit_at_3 = dy.Float64(min=0.0, max=1.0)
    hit_at_10 = dy.Float64(min=0.0, max=1.0)
    mrr = dy.Float64(min=0.0, max=1.0)
    ndcg_at_10 = dy.Float64(min=0.0, max=1.0)
    median_rank = dy.Float64(nullable=True, min=1.0, max=10.0)
    not_found_pct = dy.Float64(min=0.0, max=1.0)


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


def rank_by_score(*over: str) -> pl.Expr:
    """`rank`: 1 for the highest `score` within each group of *over*."""
    return pl.col("score").rank("ordinal", descending=True).over(*over).cast(pl.UInt8).alias("rank")


def at_target_path() -> pl.Expr:
    """Whether a candidate covers the query's target file.

    A candidate carries every path its content sits at (`file_paths`),
    so the target's `file_path` counts as found when it is one of them.
    """
    return pl.col("file_paths").list.contains(pl.col("file_path"))
