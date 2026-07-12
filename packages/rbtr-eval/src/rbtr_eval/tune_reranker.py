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
from rbtr_eval.agg import search_metric_aggs
from rbtr_eval.formatting import md_table
from rbtr_eval.queries import load_all_queries, sample_distribution, subsample, with_query_kind
from rbtr_eval.rbtr_cli import daemon_session
from rbtr_eval.schemas import QueryMeta, QueryRow, RerankerCandidate

# ── Candidate collection ────────────────────────────────────────────────────


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
                        "file_path": r.file_path,
                        "scope": r.scope,
                        "name": r.name,
                        "line_start": r.line_start,
                        "fusion": signals.fusion,
                        "reranker": signals.reranker,
                        "latency_ms": latency_ms,
                    }
                )

            done += 1
            if on_progress is not None:
                on_progress(done, total)

    candidates = pl.DataFrame(rows, schema=RerankerCandidate.to_polars_schema()).pipe(
        RerankerCandidate.validate, cast=True
    )

    meta = (
        with_query_kind(queries)
        .with_row_index("query_idx")
        .select(
            "query_idx",
            "slug",
            "language",
            "provenance",
            "query_kind",
            "file_path",
            "scope",
            "name",
            "line_start",
        )
        .pipe(QueryMeta.validate, cast=True)
    )

    return candidates, meta


# ── Offline blend + ranking ──────────────────────────────────────────────────


def _rank_all_blends(
    candidates: dy.DataFrame[RerankerCandidate],
    meta: dy.DataFrame[QueryMeta],
    blend_values: list[float],
) -> pl.DataFrame:
    """Re-blend at every blend weight and find target ranks.

    Cross-joins candidates with the blend grid, computes
    blended scores, ranks within `(pool, blend_weight,
    query_idx)`, and joins back to `meta` to find each
    query's target rank.

    Returns one row per `(pool, blend_weight, query_idx)` with
    columns: `pool, blend_weight, slug, language, provenance,
    query_kind, rank, latency_ms`.
    """
    blend_frame = pl.DataFrame(
        {"blend_weight": blend_values},
        schema={"blend_weight": pl.Float64},
    )

    scored = (
        candidates.join(blend_frame, how="cross")
        .with_columns(
            (
                pl.col("blend_weight") * pl.col("fusion")
                + (1.0 - pl.col("blend_weight")) * pl.col("reranker")
            ).alias("score"),
        )
        .with_columns(
            pl.col("score")
            .rank("ordinal", descending=True)
            .over("pool", "blend_weight", "query_idx")
            .cast(pl.UInt8)
            .alias("rank"),
        )
        .filter(pl.col("rank") <= 10)
    )

    # Find target rank: join on the identity columns.
    target_ranks = (
        scored.join(
            meta.select("query_idx", "file_path", "scope", "name", "line_start"),
            on=["query_idx", "file_path", "scope", "name", "line_start"],
            how="inner",
        )
        .group_by("pool", "blend_weight", "query_idx")
        .agg(pl.col("rank").min().alias("rank"))
    )

    # Every (pool, blend, query) must appear — build the
    # full key set and left-join target ranks.
    pool_query = candidates.select(
        "pool",
        "query_idx",
        "latency_ms",
    ).unique(subset=["pool", "query_idx"])

    all_keys = pool_query.join(blend_frame, how="cross")

    return (
        all_keys.join(
            target_ranks,
            on=["pool", "blend_weight", "query_idx"],
            how="left",
        )
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
    )


# ── Aggregation ──────────────────────────────────────────────────────────────


def _aggregate_grid(results: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per (pool, blend_weight) following `measure._aggregate`.

    Uses `search_metric_aggs` for rank-based metrics plus
    latency quantiles — same columns as `Metrics`.
    """
    latency = pl.col("latency_ms")
    return (
        results.group_by("pool", "blend_weight")
        .agg(
            *search_metric_aggs(),
            latency.quantile(0.5).alias("search_p50_ms"),
            latency.quantile(0.95).alias("search_p95_ms"),
        )
        .sort("mrr", "pool", descending=[True, False])
    )


def _aggregate_by_dimension(results: pl.DataFrame, key: str) -> pl.DataFrame:
    """Aggregate per `(pool, blend_weight, key)`, carrying p50 latency."""
    return (
        results.group_by("pool", "blend_weight", key)
        .agg(
            *search_metric_aggs(),
            pl.col("latency_ms").quantile(0.5).alias("search_p50_ms"),
        )
        .sort(key, "mrr", "pool", descending=[False, True, False])
    )


# ── Report rendering ────────────────────────────────────────────────────────


_TOLERANCES = (0.005, 0.01, 0.02, 0.03, 0.05)


def _cheapest_within_tolerance(configs: pl.DataFrame, tol: float) -> dict[str, float | int]:
    """Smallest-pool config whose MRR is within `tol` (relative) of the best.

    `configs` holds one row per `(pool, blend_weight)` with `mrr` and
    `search_p50_ms`.  Latency is monotonic in pool, so the smallest pool
    clearing the threshold is the fastest acceptable config; ties within
    a pool break to the highest MRR (best blend).
    """
    best = float(configs.select(pl.col("mrr").max()).item())
    chosen = (
        configs.filter(pl.col("mrr") >= best * (1.0 - tol))
        .sort("pool", "mrr", descending=[False, True])
        .row(0, named=True)
    )
    return {
        "pool": chosen["pool"],
        "blend_weight": chosen["blend_weight"],
        "mrr": chosen["mrr"],
        "search_p50_ms": chosen["search_p50_ms"],
        "mrr_forgone": best - chosen["mrr"],
    }


def _tolerance_sweep(by_kind: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """Cheapest pool within each tolerance, per query kind and overall.

    Surfaces the MRR/latency trade-off so the pool is chosen manually.
    Rows: `(scope, tolerance_pct, pool, blend_weight, mrr,
    search_p50_ms, mrr_forgone)`.
    """
    scopes: list[tuple[str, pl.DataFrame]] = [
        (kind, by_kind.filter(pl.col("query_kind") == kind))
        for kind in by_kind.get_column("query_kind").unique().sort().to_list()
    ]
    scopes.append(("all", grid))
    rows: list[dict[str, float | int | str]] = [
        {"scope": scope, "tolerance_pct": tol * 100.0, **_cheapest_within_tolerance(configs, tol)}
        for scope, configs in scopes
        for tol in _TOLERANCES
    ]
    return pl.DataFrame(rows)


def _render_report(
    grid: pl.DataFrame,
    by_provenance: pl.DataFrame,
    by_kind: pl.DataFrame,
    by_slug: pl.DataFrame,
    by_language: pl.DataFrame,
    dist: pl.DataFrame,
    *,
    n_queries: int,
    elapsed_seconds: float,
) -> str:
    """Render the tuning report from the aggregated grid.

    All formatting is done with polars expressions; the
    template receives pre-rendered markdown table strings.
    Follows the same pattern as `tune._render_tuning_report`.
    """
    # ── Grid table (all configs) ─────────────────────────────
    grid_display = grid.select(
        pl.col("pool"),
        pl.col("blend_weight").alias("blend"),
        pl.col("n_queries"),
        pl.col("mrr").round(4).alias("MRR"),
        pl.col("ndcg_at_10").round(4).alias("NDCG@10"),
        (pl.col("hit_at_1") * 100).round(1).alias("hit@1 %"),
        (pl.col("hit_at_3") * 100).round(1).alias("hit@3 %"),
        (pl.col("hit_at_10") * 100).round(1).alias("hit@10 %"),
        (pl.col("not_found_pct") * 100).round(1).alias("miss %"),
        pl.col("search_p50_ms").round(0).cast(pl.Int64).alias("p50 ms"),
        pl.col("search_p95_ms").round(0).cast(pl.Int64).alias("p95 ms"),
    )

    # ── Best config TOML ─────────────────────────────────────
    best = grid.row(0, named=True)

    # ── Run metadata table ───────────────────────────────────
    run_meta = pl.DataFrame(
        {
            "field": ["queries evaluated", "configs evaluated", "elapsed"],
            "value": [
                str(n_queries),
                str(grid.height),
                f"{elapsed_seconds:.0f} s",
            ],
        }
    )

    template = resources.files("rbtr_eval.templates").joinpath("reranker_tuning.md.j2").read_text()

    # ── Per-provenance table ──────────────────────────────
    prov_display = by_provenance.select(
        pl.col("provenance"),
        pl.col("pool"),
        pl.col("blend_weight").alias("blend"),
        pl.col("n_queries"),
        pl.col("mrr").round(4).alias("MRR"),
        pl.col("ndcg_at_10").round(4).alias("NDCG@10"),
        (pl.col("hit_at_1") * 100).round(1).alias("hit@1 %"),
        (pl.col("hit_at_3") * 100).round(1).alias("hit@3 %"),
    )

    # ── Per-kind table (all configs) ───────────────────────────
    kind_display = by_kind.select(
        pl.col("query_kind").alias("kind"),
        pl.col("pool"),
        pl.col("blend_weight").alias("blend"),
        pl.col("n_queries"),
        pl.col("mrr").round(4).alias("MRR"),
        pl.col("ndcg_at_10").round(4).alias("NDCG@10"),
        (pl.col("hit_at_1") * 100).round(1).alias("hit@1 %"),
        (pl.col("hit_at_3") * 100).round(1).alias("hit@3 %"),
        pl.col("search_p50_ms").round(0).cast(pl.Int64).alias("p50 ms"),
    )

    # ── Latency-aware tolerance sweep (manual pick) ──────
    tolerance_display = _tolerance_sweep(by_kind, grid).select(
        pl.col("scope").alias("kind"),
        pl.col("tolerance_pct").round(1).alias("tol %"),
        pl.col("pool"),
        pl.col("blend_weight").alias("blend"),
        pl.col("mrr").round(4).alias("MRR"),
        pl.col("search_p50_ms").round(0).cast(pl.Int64).alias("p50 ms"),
        pl.col("mrr_forgone").round(4).alias("MRR vs best"),
    )

    # ── Per-slug / per-language tables (best config only) ──
    best_pool = best["pool"]
    best_blend = best["blend_weight"]

    def _filter_best(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            (pl.col("pool") == best_pool) & ((pl.col("blend_weight") - best_blend).abs() < 0.01)
        )

    def _dimension_display(df: pl.DataFrame, key: str) -> pl.DataFrame:
        return _filter_best(df).select(
            pl.col(key),
            pl.col("n_queries"),
            pl.col("mrr").round(4).alias("MRR"),
            pl.col("ndcg_at_10").round(4).alias("NDCG@10"),
            (pl.col("hit_at_1") * 100).round(1).alias("hit@1 %"),
            (pl.col("hit_at_3") * 100).round(1).alias("hit@3 %"),
        )

    return minijinja.Environment().render_str(
        template,
        grid_table=md_table(grid_display),
        provenance_table=md_table(prov_display),
        kind_table=md_table(kind_display),
        tolerance_table=md_table(tolerance_display),
        slug_table=md_table(_dimension_display(by_slug, "slug")),
        language_table=md_table(_dimension_display(by_language, "language")),
        sample_table=md_table(dist),
        best_pool=best_pool,
        best_blend=best_blend,
        run_table=md_table(run_meta),
    )


# ── Grid generation ──────────────────────────────────────────────────────────


def _pool_grid(pool_min: int, pool_max: int, pool_step: int) -> list[int]:
    """Generate pool values from scalar params."""
    return list(range(pool_min, pool_max + 1, pool_step))


def _blend_grid(blend_min: float, blend_max: float, blend_steps: int) -> list[float]:
    """Generate evenly-spaced blend values in [blend_min, blend_max]."""
    if blend_steps < 2:
        return [round((blend_min + blend_max) / 2, 4)]
    span = blend_max - blend_min
    return [round(blend_min + i * span / (blend_steps - 1), 4) for i in range(blend_steps)]


# ── Entry point ──────────────────────────────────────────────────────────────


class TuneRerankerCmd(BaseModel):
    """Grid-search reranker pool size and blend weight."""

    per_repo_dir: Path = Field(description="Directory holding per-repo parquet files.")
    concept_dir: Path = Field(description="Directory holding concept parquet files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    data_dir: Path = Field(description="Directory for the DuckDB index.")
    config_dir: Path = Field(description="Directory for config.")
    log_dir: Path = Field(description="Directory for logs.")
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
            daemon_session(
                self.data_dir,
                self.config_dir,
                self.log_dir,
                recv_timeout_ms=120_000,
            ) as client,
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
        grid = _aggregate_grid(ranked)
        by_provenance = _aggregate_by_dimension(ranked, "provenance")
        by_kind = _aggregate_by_dimension(ranked, "query_kind")
        by_slug = _aggregate_by_dimension(ranked, "slug")
        by_language = _aggregate_by_dimension(ranked, "language")
        dist = sample_distribution(
            queries,
            strat_keys=("slug", "language", "provenance"),
        )

        report_text = _render_report(
            grid,
            by_provenance,
            by_kind,
            by_slug,
            by_language,
            dist,
            n_queries=queries.height,
            elapsed_seconds=elapsed,
        )

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(report_text, encoding="utf-8")
