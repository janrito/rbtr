"""`rbtr-eval tune-reranker` subcommand.

Grid-search `reranker_pool` and `reranker_blend_weight`
against a subsampled query set.

Pool requires a daemon call per query (it controls how many
fusion candidates enter the reranker).  Blend is varied
offline from the `fusion` and `reranker` scores the daemon
returns — re-blending is a single polars expression over
fields already in the response, not a reimplementation of
model logic.
"""

from __future__ import annotations

import time
from importlib import resources
from pathlib import Path

import dataframely as dy
import minijinja
import polars as pl
from pydantic import BaseModel, Field

from rbtr.cli.output import ProgressCallback, progress_reporter
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import SearchRequest, SearchResponse
from rbtr.domain.models import ChunkKind
from rbtr_eval.agg import (
    Dimension,
    SearchMetrics,
    at_target_path,
    rank_by_score,
    search_metric_aggs,
)
from rbtr_eval.formatting import md_table
from rbtr_eval.queries import load_all_queries, query_kind, sample_distribution, subsample
from rbtr_eval.rbtr_cli import daemon_session
from rbtr_eval.shared_schemas import (
    IDENTITY_COLUMNS,
    MATCH_COLUMNS,
    QueryMeta,
    QueryRow,
)


class RerankerCandidate(dy.Schema):
    """One row per (pool, query, result) from the daemon.

    Produced by `tune_reranker._collect_candidates`;
    consumed by `tune_reranker._rank_all_blends`.
    """

    pool = dy.Int64(min=1)
    query_idx = dy.UInt32()
    file_paths = dy.List(dy.String())
    scope = dy.String()
    name = dy.String()
    line_start = dy.UInt32()
    line_end = dy.UInt32()
    symbol_kind = dy.Enum(k.value for k in ChunkKind)
    fusion = dy.Float64()
    reranker = dy.Float64()
    latency_ms = dy.Float64(min=0.0)


class BlendRank(dy.Schema):
    """Each query's target rank under one `(pool, blend_weight)`; null past 10."""

    pool = dy.Int64(min=1)
    blend_weight = dy.Float64()
    slug = dy.String()
    language = dy.String()
    provenance = dy.String()
    query_kind = dy.String()
    rank = dy.UInt8(nullable=True, min=1, max=10)
    latency_ms = dy.Float64(min=0.0)


class GridMetrics(SearchMetrics):
    """Search metrics for one `(pool, blend_weight)`, with its latency quantiles."""

    pool = dy.Int64(min=1, primary_key=True)
    blend_weight = dy.Float64(primary_key=True)
    search_p50_ms = dy.Float64()
    search_p95_ms = dy.Float64()


class RerankerBreakdown(SearchMetrics):
    """`GridMetrics` over the queries whose `dimension` column holds `value`."""

    pool = dy.Int64(min=1, primary_key=True)
    blend_weight = dy.Float64(primary_key=True)
    dimension = dy.Enum((d.value for d in Dimension), primary_key=True)
    value = dy.String(primary_key=True)
    search_p50_ms = dy.Float64()


class ToleranceChoice(dy.Schema):
    """The fastest config within one MRR tolerance of a scope's best."""

    scope = dy.String()
    tolerance_pct = dy.Float64()
    pool = dy.Int64(min=1)
    blend_weight = dy.Float64()
    mrr = dy.Float64()
    search_p50_ms = dy.Float64()
    mrr_forgone = dy.Float64()


def _collect_candidates(
    client: DaemonClient,
    queries: dy.DataFrame[QueryRow],
    repos_dir: Path,
    pool_grid: list[int],
    on_progress: ProgressCallback | None = None,
) -> tuple[dy.DataFrame[RerankerCandidate], dy.DataFrame[QueryMeta]]:
    """Query the daemon once per (pool, query).

    Returns all result rows with `fusion` and `reranker`
    scores, plus a query-metadata frame for rank computation.
    Follows the same collection pattern as
    `tune._collect_scored_candidates`.
    """
    total = len(pool_grid) * queries.height
    rows = []
    done = 0

    for pool in pool_grid:
        for idx, query in enumerate(queries.iter_rows(named=True)):
            repo_path = (repos_dir / query["slug"]).resolve()
            t0 = time.monotonic()
            resp = client.send_or_raise_as(
                SearchResponse,
                SearchRequest(
                    repo_path=str(repo_path),
                    query=query["text"],
                    reranker_pool=pool,
                    explain=True,
                ),
            )
            latency_ms = (time.monotonic() - t0) * 1000.0

            for r in resp.results:
                signals = r.signals
                if signals is None:
                    msg = "search must return signals when explain=True"
                    raise RuntimeError(msg)
                rows.append(
                    {
                        "pool": pool,
                        "query_idx": idx,
                        "file_paths": r.file_paths,
                        "scope": r.scope,
                        "name": r.name,
                        "line_start": r.line_start,
                        "line_end": r.line_end,
                        "symbol_kind": r.kind.value,
                        "fusion": signals.fusion,
                        "reranker": signals.reranker,
                        "latency_ms": latency_ms,
                    }
                )

            done += 1
            if on_progress is not None:
                on_progress(done, total)

    candidates = pl.DataFrame(rows, schema=RerankerCandidate.create_empty().schema).pipe(
        RerankerCandidate.validate, cast=True
    )

    meta = (
        queries.with_row_index("query_idx")
        .select(
            "query_idx",
            "slug",
            "language",
            "provenance",
            query_kind(),
            *IDENTITY_COLUMNS,
        )
        .pipe(QueryMeta.validate, cast=True)
    )

    return candidates, meta


def _rank_all_blends(
    candidates: dy.DataFrame[RerankerCandidate],
    meta: dy.DataFrame[QueryMeta],
    blend_values: list[float],
) -> dy.DataFrame[BlendRank]:
    """Re-blend at every blend weight and find target ranks.

    Cross-joins candidates with the blend grid, ranks the blended
    scores within `(pool, blend_weight, query_idx)`, and finds each
    query's target among the top 10.  Every `(pool, blend_weight,
    query_idx)` gets a row; its rank is null when the target is not in
    the top 10.
    """
    blends = pl.DataFrame({"blend_weight": blend_values}, schema={"blend_weight": pl.Float64})
    ranks = (
        candidates.join(blends, how="cross")
        .with_columns(
            (
                pl.col("blend_weight") * pl.col("fusion")
                + (1.0 - pl.col("blend_weight")) * pl.col("reranker")
            ).alias("score"),
        )
        .with_columns(rank_by_score("pool", "blend_weight", "query_idx"))
        .filter(pl.col("rank") <= 10)
        .join(
            meta.select("query_idx", *IDENTITY_COLUMNS),
            on=["query_idx", *MATCH_COLUMNS],
            how="inner",
        )
        .filter(at_target_path())
        .group_by("pool", "blend_weight", "query_idx")
        .agg(pl.col("rank").min())
    )
    return (
        candidates.select("pool", "query_idx", "latency_ms")
        .unique(subset=["pool", "query_idx"])
        .join(blends, how="cross")
        .join(ranks, on=["pool", "blend_weight", "query_idx"], how="left")
        .join(
            meta.select("query_idx", "slug", "language", "provenance", "query_kind"),
            on="query_idx",
            how="left",
        )
        .select(
            "pool",
            "blend_weight",
            "slug",
            "language",
            "provenance",
            "query_kind",
            "rank",
            "latency_ms",
        )
        .pipe(BlendRank.validate, cast=True)
    )


def _grid_metrics(ranked: dy.DataFrame[BlendRank]) -> dy.DataFrame[GridMetrics]:
    """Search metrics per `(pool, blend_weight)`, best MRR first."""
    return (
        ranked.group_by("pool", "blend_weight")
        .agg(
            *search_metric_aggs(),
            pl.col("latency_ms").quantile(0.5).alias("search_p50_ms"),
            pl.col("latency_ms").quantile(0.95).alias("search_p95_ms"),
        )
        .sort("mrr", "pool", "blend_weight", descending=[True, False, False])
        .pipe(GridMetrics.validate, cast=True)
    )


def _breakdowns(ranked: dy.DataFrame[BlendRank]) -> dy.DataFrame[RerankerBreakdown]:
    """Search metrics per `(pool, blend_weight)` and value of each dimension.

    Within a dimension value, the best MRR comes first.
    """
    return (
        pl.concat(
            ranked.group_by("pool", "blend_weight", dimension.value)
            .agg(
                *search_metric_aggs(),
                pl.col("latency_ms").quantile(0.5).alias("search_p50_ms"),
            )
            .select(
                "pool",
                "blend_weight",
                pl.lit(dimension.value).alias("dimension"),
                pl.col(dimension.value).alias("value"),
                pl.exclude("pool", "blend_weight", dimension.value),
            )
            for dimension in (
                Dimension.PROVENANCE,
                Dimension.QUERY_KIND,
                Dimension.SLUG,
                Dimension.LANGUAGE,
            )
        )
        .sort(
            "dimension",
            "value",
            "mrr",
            "pool",
            "blend_weight",
            descending=[False, False, True, False, False],
        )
        .pipe(RerankerBreakdown.validate, cast=True)
    )


def _tolerance_sweep(
    grid: dy.DataFrame[GridMetrics], breakdowns: dy.DataFrame[RerankerBreakdown]
) -> dy.DataFrame[ToleranceChoice]:
    """The fastest config within each MRR tolerance, per query kind and overall.

    A scope's configs clear a tolerance when their MRR is within it
    (relative) of the scope's best.  Latency rises with pool, so the
    smallest pool that clears it is the fastest; ties within a pool go
    to the highest MRR, then the lowest blend weight.  Kinds come in
    name order, then `all`.
    """
    columns = ["scope", "pool", "blend_weight", "mrr", "search_p50_ms"]
    return (
        pl.concat(
            [
                breakdowns.filter(pl.col("dimension") == Dimension.QUERY_KIND)
                .rename({"value": "scope"})
                .select(columns),
                grid.with_columns(pl.lit("all").alias("scope")).select(columns),
            ]
        )
        .join(pl.DataFrame({"tolerance": [0.005, 0.01, 0.02, 0.03, 0.05]}), how="cross")
        .with_columns(pl.col("mrr").max().over("scope").alias("best"))
        .filter(pl.col("mrr") >= pl.col("best") * (1.0 - pl.col("tolerance")))
        .sort(
            pl.col("scope") == "all",
            "scope",
            "tolerance",
            "pool",
            "mrr",
            "blend_weight",
            descending=[False, False, False, False, True, False],
        )
        .group_by("scope", "tolerance", maintain_order=True)
        .first()
        .select(
            "scope",
            (pl.col("tolerance") * 100.0).alias("tolerance_pct"),
            "pool",
            "blend_weight",
            "mrr",
            "search_p50_ms",
            (pl.col("best") - pl.col("mrr")).alias("mrr_forgone"),
        )
        .pipe(ToleranceChoice.validate, cast=True)
    )


def _quality_columns() -> list[pl.Expr]:
    """The display columns every table shows: n, MRR, NDCG@10, hit@1 and hit@3."""
    return [
        pl.col("n_queries"),
        pl.col("mrr").round(4).alias("MRR"),
        pl.col("ndcg_at_10").round(4).alias("NDCG@10"),
        (pl.col("hit_at_1") * 100).round(1).alias("hit@1 %"),
        (pl.col("hit_at_3") * 100).round(1).alias("hit@3 %"),
    ]


def _render_report(
    grid: dy.DataFrame[GridMetrics],
    breakdowns: dy.DataFrame[RerankerBreakdown],
    tolerance: dy.DataFrame[ToleranceChoice],
    dist: pl.DataFrame,
    *,
    n_queries: int,
    elapsed_seconds: float,
) -> str:
    """Render the tuning report.

    The best config is the grid's first row; the per-repo and
    per-language tables show it alone.
    """
    best_pool, best_blend = grid.select("pool", "blend_weight").row(0)
    at_best = (pl.col("pool") == best_pool) & ((pl.col("blend_weight") - best_blend).abs() < 0.01)

    run_table = pl.DataFrame(
        {
            "field": ["queries evaluated", "configs evaluated", "elapsed"],
            "value": [str(n_queries), str(grid.height), f"{elapsed_seconds:.0f} s"],
        }
    )
    grid_table = grid.select(
        pl.col("pool"),
        pl.col("blend_weight").alias("blend"),
        *_quality_columns(),
        (pl.col("hit_at_10") * 100).round(1).alias("hit@10 %"),
        (pl.col("not_found_pct") * 100).round(1).alias("miss %"),
        pl.col("search_p50_ms").round(0).cast(pl.Int64).alias("p50 ms"),
        pl.col("search_p95_ms").round(0).cast(pl.Int64).alias("p95 ms"),
    )
    provenance_table = breakdowns.filter(pl.col("dimension") == Dimension.PROVENANCE).select(
        pl.col("value").alias("provenance"),
        "pool",
        pl.col("blend_weight").alias("blend"),
        *_quality_columns(),
    )
    kind_table = breakdowns.filter(pl.col("dimension") == Dimension.QUERY_KIND).select(
        pl.col("value").alias("kind"),
        "pool",
        pl.col("blend_weight").alias("blend"),
        *_quality_columns(),
        pl.col("search_p50_ms").round(0).cast(pl.Int64).alias("p50 ms"),
    )
    tolerance_table = tolerance.select(
        pl.col("scope").alias("kind"),
        pl.col("tolerance_pct").round(1).alias("tol %"),
        pl.col("pool"),
        pl.col("blend_weight").alias("blend"),
        pl.col("mrr").round(4).alias("MRR"),
        pl.col("search_p50_ms").round(0).cast(pl.Int64).alias("p50 ms"),
        pl.col("mrr_forgone").round(4).alias("MRR vs best"),
    )
    slug_table = breakdowns.filter(pl.col("dimension") == Dimension.SLUG, at_best).select(
        pl.col("value").alias("slug"), *_quality_columns()
    )
    language_table = breakdowns.filter(pl.col("dimension") == Dimension.LANGUAGE, at_best).select(
        pl.col("value").alias("language"), *_quality_columns()
    )

    template = resources.files("rbtr_eval.templates").joinpath("reranker_tuning.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        grid_table=md_table(grid_table),
        provenance_table=md_table(provenance_table),
        kind_table=md_table(kind_table),
        tolerance_table=md_table(tolerance_table),
        slug_table=md_table(slug_table),
        language_table=md_table(language_table),
        sample_table=md_table(dist),
        best_pool=best_pool,
        best_blend=best_blend,
        run_table=md_table(run_table),
    )


def _pool_grid(pool_min: int, pool_max: int, pool_step: int) -> list[int]:
    """Generate pool values from scalar params."""
    return list(range(pool_min, pool_max + 1, pool_step))


def _blend_grid(blend_min: float, blend_max: float, blend_steps: int) -> list[float]:
    """Generate evenly-spaced blend values in [blend_min, blend_max]."""
    if blend_steps < 2:
        return [round((blend_min + blend_max) / 2, 4)]
    span = blend_max - blend_min
    return [round(blend_min + i * span / (blend_steps - 1), 4) for i in range(blend_steps)]


class TuneRerankerCmd(BaseModel):
    """Grid-search reranker pool size and blend weight."""

    per_repo_dir: Path = Field(description="Directory holding per-repo parquet files.")
    concept_dir: Path = Field(description="Directory holding concept parquet files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    queries_per_cell: int = Field(
        2, description="Queries per (slug, language, provenance) cell for tuning."
    )
    pool_min: int = Field(10, description="Smallest pool size.")
    pool_max: int = Field(50, description="Largest pool size.")
    pool_step: int = Field(10, description="Pool size step.")
    blend_min: float = Field(0.2, description="Smallest blend weight.")
    blend_max: float = Field(0.8, description="Largest blend weight.")
    blend_steps: int = Field(5, description="Number of evenly-spaced blend values.")
    seed: int = Field(0, description="Deterministic RNG seed for subsampling.")
    report: Path = Field(description="Output path for RERANKER_TUNING.md.")

    def cli_cmd(self) -> None:
        all_queries = load_all_queries(self.per_repo_dir, self.concept_dir)

        queries = subsample(
            all_queries,
            queries_per_cell=self.queries_per_cell,
            seed=self.seed,
            strat_keys=("slug", "language", "provenance"),
        )

        pools = _pool_grid(self.pool_min, self.pool_max, self.pool_step)
        blends = _blend_grid(self.blend_min, self.blend_max, self.blend_steps)

        t0 = time.monotonic()

        with (
            daemon_session() as client,
            progress_reporter("tune-reranker") as (on_progress,),
        ):
            candidates, meta = _collect_candidates(
                client,
                queries,
                self.repos_dir,
                pools,
                on_progress=on_progress,
            )

        ranked = _rank_all_blends(candidates, meta, blends)
        elapsed = time.monotonic() - t0
        grid = _grid_metrics(ranked)
        breakdowns = _breakdowns(ranked)

        report_text = _render_report(
            grid,
            breakdowns,
            _tolerance_sweep(grid, breakdowns),
            sample_distribution(queries, strat_keys=("slug", "language", "provenance")),
            n_queries=queries.height,
            elapsed_seconds=elapsed,
        )

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(report_text, encoding="utf-8")
