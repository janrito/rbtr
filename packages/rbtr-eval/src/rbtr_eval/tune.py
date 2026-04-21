"""`rbtr-eval tune` subcommand.

Grid-search the rbtr search fusion weights `(alpha, beta,
gamma)` against every per-repo query set, using the
full-variant index in the shared home.  Reports best vs
current weights in `data/tuned-params.json`; never edits
source.

Indexing is a separate DVC stage; this command only queries.
One warm daemon serves every grid point for every query.
Ranking uses the same declarative polars explode + `int_range`
pattern as `measure`; aggregation is one
`group_by([alpha, beta, gamma]).agg(mrr)` + `head(1)`.
"""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path

import dataframely as dy
import polars as pl
from pydantic import BaseModel, Field

from rbtr.config import config as rbtr_config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import SearchRequest, SearchResponse
from rbtr.git import read_head
from rbtr.index.models import IndexVariant
from rbtr_eval.rbtr_cli import daemon_session
from rbtr_eval.schemas import (
    QueryRow,
    TuneReport,
    WeightedSearchBatch,
    WeightedSearchOutcome,
)


def grid_triples(step: float) -> list[tuple[float, float, float]]:
    """Enumerate `(alpha, beta, gamma)` in `[0, 1]^3` summing to 1 at *step*.

    Uses the stars-and-bars bijection between 3-part
    compositions of `n = 1/step` and 2-subsets of
    `{0, ..., n + 1}`: picking two bar positions partitions
    the `n` stars into three groups whose lengths `(a, b, g)`
    are the composition.  `itertools.combinations` walks the
    2-subsets in lexicographic order; multiplying by `step`
    lands the triple on the unit simplex.  Values are
    rounded to six decimals so equality compares cleanly.

    Yields `C(n + 2, 2) = (n + 1)(n + 2) / 2` triples: 6 at
    step 0.5, 21 at 0.2, 66 at 0.1, 231 at 0.05.  Pure
    function; exposed for tests.
    """
    if not 0.0 < step <= 1.0:
        msg = f"grid_step must be in (0, 1]; got {step}"
        raise ValueError(msg)
    n = round(1.0 / step)
    return [
        (
            round(first_bar * step, 6),
            round((second_bar - first_bar - 1) * step, 6),
            round((n + 1 - second_bar) * step, 6),
        )
        for first_bar, second_bar in combinations(range(n + 2), 2)
    ]


# ── Typed search ─────────────────────────────────────────────────────────────


def _search(
    client: DaemonClient,
    repo_path: Path,
    query: str,
    weights: tuple[float, float, float] | None,
) -> list[dict[str, str | int]]:
    """One search via the daemon; None *weights* uses config defaults."""
    a = b = g = None
    if weights is not None:
        a, b, g = weights
    request = SearchRequest(
        repo=str(repo_path),
        query=query,
        variant=IndexVariant.FULL,
        limit=10,
        alpha=a,
        beta=b,
        gamma=g,
    )
    response = client.send_or_raise_as(SearchResponse, request)
    return [
        {
            "file_path": h.chunk.file_path,
            "scope": h.chunk.scope,
            "name": h.chunk.name,
            "line_start": h.chunk.line_start,
        }
        for h in response.results
    ]


# ── Search execution ─────────────────────────────────────────────────────────


def _run_weight_trials(
    client: DaemonClient,
    queries: dy.DataFrame[QueryRow],
    repos_dir: Path,
    triples: list[tuple[float, float, float]],
) -> dy.DataFrame[WeightedSearchBatch]:
    """Run one baseline + every grid-triple search for every query.

    The weight configurations are a flat sequence: one
    `(label, weights)` for baseline plus one per grid triple.
    The inner loop treats every config the same way; baseline
    is just `weights=None`.  Returns a raw frame with
    `hits: list[struct]`; `_score_trials` expands that into
    `WeightedSearchOutcome` rows with declarative ranking.
    """
    configs: list[tuple[str, tuple[float, float, float] | None]] = [
        ("baseline", None),
        *(("grid", triple) for triple in triples),
    ]
    rows: list[dict[str, str | float | None | list[dict[str, str | int]]]] = []
    for query in queries.iter_rows(named=True):
        repo_path = (repos_dir / query["slug"]).resolve()
        for label, weights in configs:
            alpha, beta, gamma = (None, None, None) if weights is None else weights
            rows.append(
                {
                    "slug": query["slug"],
                    "label": label,
                    "query_file": query["file_path"],
                    "query_scope": query["scope"],
                    "query_name": query["name"],
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "hits": _search(client, repo_path, query["text"], weights),
                }
            )
    return pl.DataFrame(rows).pipe(WeightedSearchBatch.validate, cast=True)


def _score_trials(batch: dy.DataFrame[WeightedSearchBatch]) -> dy.DataFrame[WeightedSearchOutcome]:
    """Expand raw hits into ranked rows for every (label, triple, query).

    Same explode + `int_range().over()` + filter + join
    pattern as `measure._score_outcomes`, but keyed on the
    weight-trial identity columns.
    """
    trial_keys = [
        "slug",
        "label",
        "alpha",
        "beta",
        "gamma",
        "query_file",
        "query_scope",
        "query_name",
    ]
    exploded = (
        batch.select(*trial_keys, "hits")
        .explode("hits")
        .with_columns(
            pl.col("hits").struct.field("file_path").alias("hit_file_path"),
            pl.col("hits").struct.field("scope").alias("hit_scope"),
            pl.col("hits").struct.field("name").alias("hit_name"),
            pl.int_range(1, pl.len() + 1, dtype=pl.UInt8).over(trial_keys).alias("hit_rank"),
        )
    )
    ranks = (
        exploded.filter(
            (pl.col("hit_file_path") == pl.col("query_file"))
            & (pl.col("hit_scope") == pl.col("query_scope"))
            & (pl.col("hit_name") == pl.col("query_name"))
        )
        .group_by(trial_keys)
        .agg(pl.col("hit_rank").min().alias("rank"))
    )
    return (
        batch.drop("hits")
        # `nulls_equal=True`: baseline rows have null alpha/beta/
        # gamma, and polars' default join treats null != null,
        # which drops every baseline rank.
        .join(ranks, on=trial_keys, how="left", nulls_equal=True)
        .pipe(WeightedSearchOutcome.validate, cast=True)
    )


# ── Aggregation ──────────────────────────────────────────────────────────────


def _mrr_expr() -> pl.Expr:
    """MRR expression treating null ranks as reciprocal 0.

    Shared between the baseline and grid-best queries so both
    compute MRR the same way.
    """
    rank = pl.col("rank")
    return (
        pl.when(rank.is_null()).then(0.0).otherwise(1.0 / rank.cast(pl.Float64)).mean().alias("mrr")
    )


# ── Entry point ──────────────────────────────────────────────────────────────


class TuneCmd(BaseModel):
    """Grid-search rbtr's fusion weights against the query set."""

    per_repo_dir: Path = Field(description="Directory holding per-repo parquet files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    home: Path = Field(description="Single RBTR_HOME with the `full` index built.")
    grid_step: float = Field(0.2, description="Step size for the (alpha, beta, gamma) grid.")
    output: Path = Field(description="Output path for the tuning suggestion JSON.")

    def cli_cmd(self) -> None:
        rbtr_sha = read_head(".") or "unknown"
        queries = pl.read_parquet(self.per_repo_dir / "*.queries.parquet").pipe(
            QueryRow.validate, cast=True
        )
        triples = grid_triples(self.grid_step)
        t0 = time.monotonic()

        with daemon_session(self.home) as client:
            batch = _run_weight_trials(client, queries, self.repos_dir, triples)

        trials = _score_trials(batch)
        grid_best = (
            trials.filter(pl.col("label") == "grid")
            .group_by(["alpha", "beta", "gamma"])
            .agg(_mrr_expr())
            .sort("mrr", descending=True)
            .head(1)
        )
        baseline = trials.filter(pl.col("label") == "baseline").select(
            _mrr_expr(), pl.len().cast(pl.UInt32).alias("n_queries")
        )
        if grid_best.is_empty() or baseline.is_empty():
            msg = "no rows produced; dataset empty?"
            raise SystemExit(msg)

        report = (
            grid_best.rename(
                {
                    "alpha": "best_alpha",
                    "beta": "best_beta",
                    "gamma": "best_gamma",
                    "mrr": "score_best",
                }
            )
            .with_columns(
                pl.lit(rbtr_config.search_alpha).alias("current_alpha"),
                pl.lit(rbtr_config.search_beta).alias("current_beta"),
                pl.lit(rbtr_config.search_gamma).alias("current_gamma"),
                pl.lit(baseline["mrr"][0]).alias("score_current"),
                (pl.lit(grid_best["mrr"][0]) - pl.lit(baseline["mrr"][0])).alias("delta"),
                pl.lit("MRR").alias("metric"),
                pl.lit(self.grid_step).alias("grid_step"),
                pl.lit(baseline["n_queries"][0]).cast(pl.UInt32).alias("n_queries"),
                pl.lit(len(triples)).cast(pl.UInt32).alias("n_grid_points"),
                pl.lit(rbtr_sha).alias("rbtr_sha"),
                pl.lit(time.monotonic() - t0).alias("elapsed_seconds"),
            )
            .pipe(TuneReport.validate, cast=True)
        )

        self.output.parent.mkdir(parents=True, exist_ok=True)
        report.write_json(self.output)
