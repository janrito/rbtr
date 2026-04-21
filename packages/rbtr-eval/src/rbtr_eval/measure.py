"""`rbtr-eval measure` subcommand.

Reads per-repo parquet query files, runs every query through
the already-built rbtr index home (one shared home across all
repos and variants), and writes:

* `data/BENCHMARKS.md` - human-readable report.
* `data/metrics.json`  - DVC metrics (polars-written array).

Indexing is a separate DVC stage; this command only queries.
Searches go through `DaemonClient` against a daemon bound to
the shared home.  Ranking is declarative polars (explode +
`int_range().over()` + filter + join); aggregation is one
`group_by(...).agg(*aggs)` call per grouping level.
"""

from __future__ import annotations

import time
from importlib import resources
from pathlib import Path

import dataframely as dy
import duckdb
import minijinja
import polars as pl
from pydantic import BaseModel, Field

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import SearchRequest, SearchResponse
from rbtr.git import read_head
from rbtr.index.models import IndexVariant
from rbtr_eval.rbtr_cli import daemon_session
from rbtr_eval.schemas import (
    Metrics,
    MetricsFile,
    MissCandidate,
    QueryRow,
    RepoHeader,
    SearchBatch,
    SearchOutcome,
)

# ── Typed search ─────────────────────────────────────────────────────────────


def _search(
    client: DaemonClient,
    repo_path: Path,
    query: str,
    variant: IndexVariant,
) -> tuple[list[dict[str, str | int]], float]:
    """One search call via the daemon client; returns (hit-dicts, wall_ms)."""
    request = SearchRequest(
        repo=str(repo_path),
        query=query,
        variant=variant,
        limit=10,
    )
    t0 = time.monotonic()
    response = client.send_or_raise_as(SearchResponse, request)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    return [
        {
            "file_path": hit.chunk.file_path,
            "scope": hit.chunk.scope,
            "name": hit.chunk.name,
            "line_start": hit.chunk.line_start,
        }
        for hit in response.results
    ], elapsed_ms


# ── Search execution ─────────────────────────────────────────────────────────


def _run_searches(
    client: DaemonClient,
    queries: dy.DataFrame[QueryRow],
    repos_dir: Path,
) -> dy.DataFrame[SearchBatch]:
    """Run every `(query, variant)` search; capture hits + latency.

    Returns an un-scored outcome frame with a `hits: list[struct]`
    column.  `_score_outcomes` expands that into typed
    `SearchOutcome` rows with declarative ranking.  The loop
    is over Python inputs (query rows from parquet, enum
    variants); each row dict is one call's outcome.
    """
    rows: list[dict[str, str | float | list[dict[str, str | int]]]] = []
    for query in queries.iter_rows(named=True):
        repo_path = (repos_dir / query["slug"]).resolve()
        for variant in IndexVariant:
            hits, latency_ms = _search(client, repo_path, query["text"], variant)
            rows.append(
                {
                    "slug": query["slug"],
                    "variant": variant.value,
                    "query_file": query["file_path"],
                    "query_scope": query["scope"],
                    "query_name": query["name"],
                    "query_text": query["text"],
                    "latency_ms": latency_ms,
                    "hits": hits,
                }
            )
    return pl.DataFrame(rows).pipe(SearchBatch.validate, cast=True)


def _score_outcomes(batch: dy.DataFrame[SearchBatch]) -> dy.DataFrame[SearchOutcome]:
    """Expand raw hits into ranked + top-hit columns.

    Explodes `hits`, numbers rows within each outcome via
    `int_range().over(...)`, picks the rank of the matching
    target (if any), and the top-1 hit for the misses
    appendix.  Left-joins back so queries whose target never
    appeared keep a null rank.
    """
    outcome_keys = ["slug", "variant", "query_file", "query_scope", "query_name"]
    exploded = (
        batch.select(
            *outcome_keys,
            pl.col("hits"),
        )
        .explode("hits")
        .with_columns(
            pl.col("hits").struct.field("file_path").alias("hit_file_path"),
            pl.col("hits").struct.field("scope").alias("hit_scope"),
            pl.col("hits").struct.field("name").alias("hit_name"),
            pl.col("hits").struct.field("line_start").alias("hit_line_start"),
            pl.int_range(1, pl.len() + 1, dtype=pl.UInt8).over(outcome_keys).alias("hit_rank"),
        )
    )

    ranks = (
        exploded.filter(
            (pl.col("hit_file_path") == pl.col("query_file"))
            & (pl.col("hit_scope") == pl.col("query_scope"))
            & (pl.col("hit_name") == pl.col("query_name"))
        )
        .group_by(outcome_keys)
        .agg(pl.col("hit_rank").min().alias("rank"))
    )

    tops = exploded.filter(pl.col("hit_rank") == 1).select(
        *outcome_keys,
        pl.col("hit_file_path").alias("top_file"),
        pl.col("hit_line_start").alias("top_line"),
        pl.col("hit_name").alias("top_name"),
    )

    return (
        batch.drop("hits")
        .join(ranks, on=outcome_keys, how="left")
        .join(tops, on=outcome_keys, how="left")
        .pipe(SearchOutcome.validate, cast=True)
    )


# ── Aggregation ──────────────────────────────────────────────────────────────


def _aggregate(outcomes: dy.DataFrame[SearchOutcome]) -> dy.DataFrame[Metrics]:
    """Per-(slug, variant) metrics plus an `__all__` rollup per variant.

    Hit@k counts null ranks as misses (`fill_null(False)`).
    MRR treats null ranks as reciprocal 0 via an explicit
    `when / then / otherwise` (so `1 / null = null` doesn't
    trigger polars' default null-skipping in `mean`).
    `median_rank` drops nulls before medianing; a variant
    where every query missed returns null, which the renderer
    prints as "-".
    """
    rank = pl.col("rank")
    latency = pl.col("latency_ms")
    reciprocal = pl.when(rank.is_null()).then(0.0).otherwise(1.0 / rank.cast(pl.Float64))
    aggs = [
        pl.len().cast(pl.UInt32).alias("n_queries"),
        (rank <= 1).fill_null(False).mean().alias("hit_at_1"),
        (rank <= 3).fill_null(False).mean().alias("hit_at_3"),
        (rank <= 10).fill_null(False).mean().alias("hit_at_10"),
        reciprocal.mean().alias("mrr"),
        rank.drop_nulls().cast(pl.Float64).median().alias("median_rank"),
        rank.is_null().mean().alias("not_found_pct"),
        latency.quantile(0.5).alias("search_p50_ms"),
        latency.quantile(0.95).alias("search_p95_ms"),
    ]
    per_pair = outcomes.group_by(["slug", "variant"]).agg(*aggs)
    rollup = (
        outcomes.group_by("variant")
        .agg(*aggs)
        .with_columns(pl.lit("__all__").alias("slug"))
        .select(per_pair.columns)
    )
    return (
        pl.concat([per_pair, rollup], how="vertical")
        .sort(["slug", "variant"])
        .pipe(Metrics.validate, cast=True)
    )


# ── Notable misses ───────────────────────────────────────────────────────────


def _select_misses(
    outcomes: dy.DataFrame[SearchOutcome], limit: int = 20
) -> dy.DataFrame[MissCandidate]:
    """Top *limit* queries by largest stripped-vs-full rank gap.

    `fill_null(11)` converts "no top-10 hit" into a sentinel
    worse than any real rank, so the subtraction is always
    defined.  Queries where stripped did the same or better
    than full are filtered out (`gap > 0`).
    """
    return (
        outcomes.pivot(
            on="variant",
            index=["slug", "query_file", "query_scope", "query_name", "query_text"],
            values=["rank", "top_file", "top_line", "top_name"],
        )
        .with_columns(
            (pl.col("rank_stripped").fill_null(11) - pl.col("rank_full").fill_null(11))
            .cast(pl.Int16)
            .alias("gap")
        )
        .filter(pl.col("gap") > 0)
        .sort(["gap", "slug", "query_file"], descending=[True, False, False])
        .head(limit)
        .pipe(MissCandidate.validate, cast=True)
    )


# ── Home size via DuckDB ─────────────────────────────────────────────────────


def _home_size_bytes(home: Path, db_name: str = "index.duckdb") -> int:
    """Bytes occupied by the rbtr DuckDB index at *home*.

    Opens read-only and asks `PRAGMA database_size` — DuckDB
    reports block size, total blocks, and WAL size directly.
    Safe to call only after the daemon has stopped (DuckDB
    takes a process-level lock).
    """
    db_path = home / db_name
    if not db_path.exists():
        return 0
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("PRAGMA database_size").fetchone()
    finally:
        con.close()
    if row is None:
        return 0
    cols = ["database_name", "database_size", "block_size", "total_blocks"]
    data = dict(zip(cols, row[: len(cols)], strict=False))
    return int(data["block_size"]) * int(data["total_blocks"])


# ── Rendering ────────────────────────────────────────────────────────────────


def _md(df: pl.DataFrame) -> str:
    """Render *df* as a markdown table string via `pl.Config`."""
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        fmt_str_lengths=200,
        tbl_width_chars=10_000,
        tbl_rows=-1,
        tbl_cols=-1,
    ):
        return str(df)


def _bytes_human(n: int) -> str:
    """Human-readable byte count (B / KiB / MiB / GiB)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MiB"
    return f"{n / (1024 * 1024 * 1024):.1f} GiB"


# Display-frame projections.  Each turns a typed source frame
# (Metrics, RepoHeader, MissCandidate) into a polars frame whose
# string columns render directly as a markdown table.  No
# Python-level row dicts.

_repo_display_expr = (
    pl.when(pl.col("slug") == "__all__")
    .then(pl.lit("**all repos**"))
    .otherwise(pl.format("`{}`", pl.col("slug")))
    .alias("repo")
)


def _pct_str(col: str) -> pl.Expr:
    """Polars expression: a `0..1` float column rendered as `30.0%`."""
    return ((pl.col(col) * 100).round(1).cast(pl.String) + pl.lit("%")).alias(col)


def _headline_table(metrics_df: dy.DataFrame[Metrics]) -> pl.DataFrame:
    """`Metrics` -> display frame for the headline-metrics section."""
    return metrics_df.select(
        _repo_display_expr,
        pl.col("variant"),
        pl.col("n_queries").alias("n"),
        _pct_str("hit_at_1").alias("Hit@1"),
        _pct_str("hit_at_3").alias("Hit@3"),
        _pct_str("hit_at_10").alias("Hit@10"),
        pl.col("mrr").round(3).cast(pl.String).alias("MRR"),
        pl.when(pl.col("median_rank").is_null())
        .then(pl.lit("-"))
        .otherwise(pl.col("median_rank").cast(pl.Int64).cast(pl.String))
        .alias("median rank"),
        _pct_str("not_found_pct").alias("not found"),
    )


def _latency_table(metrics_df: dy.DataFrame[Metrics]) -> pl.DataFrame:
    """`Metrics` -> display frame for the search-latency section."""
    return metrics_df.select(
        _repo_display_expr,
        pl.col("variant"),
        (pl.col("search_p50_ms").round(0).cast(pl.Int64).cast(pl.String) + pl.lit(" ms")).alias(
            "search P50"
        ),
        (pl.col("search_p95_ms").round(0).cast(pl.Int64).cast(pl.String) + pl.lit(" ms")).alias(
            "search P95"
        ),
    )


def _repos_table(headers: dy.DataFrame[RepoHeader]) -> pl.DataFrame:
    """`RepoHeader` -> display frame for the per-repo summary."""
    return headers.sort("slug").select(
        pl.format("`{}`", pl.col("slug")).alias("slug"),
        pl.format("`{}`", pl.col("sha").str.slice(0, 12)).alias("sha"),
        pl.col("n_sampled").alias("n queries"),
    )


def _render_report(
    *,
    headers: dy.DataFrame[RepoHeader],
    rbtr_sha: str,
    elapsed_seconds: float,
    metrics_df: dy.DataFrame[Metrics],
    misses_df: dy.DataFrame[MissCandidate],
    shared_home_bytes: int,
) -> str:
    """Render `BENCHMARKS.md` from the jinja template.

    Tables come from polars (`pl.Config(tbl_formatting='MARKDOWN')`)
    pre-rendered as strings.  The misses appendix is a list of
    multi-line code blocks, not tabular, so it stays as a
    Python loop in the template context.  Cosmetic markdown
    formatting (column alignment, line wrapping) is left to
    `just lint-md` / CI; this stage emits whatever polars and
    jinja produce.
    """
    template = resources.files("rbtr_eval.templates").joinpath("benchmarks.md.j2").read_text()

    misses_ctx = [
        {
            "slug": r["slug"],
            "file_path": r["query_file"],
            "scope_name": (
                f"{r['query_scope']}.{r['query_name']}" if r["query_scope"] else r["query_name"]
            ),
            "text": r["query_text"],
            "full_rank_str": "-" if r["rank_full"] is None else str(int(r["rank_full"])),
            "stripped_rank_str": (
                "-" if r["rank_stripped"] is None else str(int(r["rank_stripped"]))
            ),
            "full_top_str": (
                f"{r['top_file_full']}:{int(r['top_line_full'])} {r['top_name_full']}"
                if r["top_file_full"] is not None
                else "-"
            ),
            "stripped_top_str": (
                f"{r['top_file_stripped']}:{int(r['top_line_stripped'])} {r['top_name_stripped']}"
                if r["top_file_stripped"] is not None
                else "-"
            ),
        }
        for r in misses_df.iter_rows(named=True)
    ]

    run_table = pl.DataFrame(
        {
            "field": ["rbtr commit", "seed", "sample cap", "total queries", "elapsed"],
            "value": [
                f"`{rbtr_sha}`",
                str(int(headers["seed"][0])),
                str(int(headers["sample_cap"][0])),
                str(int(headers["n_sampled"].sum())),
                f"{round(elapsed_seconds)} s",
            ],
        }
    )

    return minijinja.Environment().render_str(
        template,
        shared_home_bytes_human=_bytes_human(shared_home_bytes),
        run_table=_md(run_table),
        repos_table=_md(_repos_table(headers)),
        headline_table=_md(_headline_table(metrics_df)),
        latency_table=_md(_latency_table(metrics_df)),
        misses=misses_ctx,
    )


# ── Entry point ──────────────────────────────────────────────────────────────


class MeasureCmd(BaseModel):
    """Replay queries against an already-built shared index home."""

    per_repo_dir: Path = Field(description="Directory holding per-repo parquet files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    home: Path = Field(description="Single RBTR_HOME holding both index variants.")
    report: Path = Field(description="Output path for BENCHMARKS.md.")
    metrics: Path = Field(description="Output path for metrics JSON.")

    def cli_cmd(self) -> None:
        rbtr_sha = read_head(".") or "unknown"
        queries = pl.read_parquet(self.per_repo_dir / "*.queries.parquet").pipe(
            QueryRow.validate, cast=True
        )
        headers = pl.read_parquet(self.per_repo_dir / "*.header.parquet").pipe(
            RepoHeader.validate, cast=True
        )

        t0 = time.monotonic()
        with daemon_session(self.home) as client:
            batch = _run_searches(client, queries, self.repos_dir)
        elapsed_seconds = time.monotonic() - t0
        shared_home_bytes = _home_size_bytes(self.home)

        outcomes = _score_outcomes(batch)
        metrics_df = _aggregate(outcomes)
        misses_df = _select_misses(outcomes)

        report_text = _render_report(
            headers=headers,
            rbtr_sha=rbtr_sha,
            elapsed_seconds=elapsed_seconds,
            metrics_df=metrics_df,
            misses_df=misses_df,
            shared_home_bytes=shared_home_bytes,
        )

        # metrics.json: metrics frame + per-slug SHA (joined) + run
        # metadata as literal columns.  `__all__` rollup rows stay
        # null on sha.
        metrics_file = (
            metrics_df.join(headers.select("slug", "sha"), on="slug", how="left")
            .with_columns(
                pl.lit(rbtr_sha).alias("rbtr_sha"),
                pl.lit(int(headers["seed"][0])).cast(pl.UInt32).alias("seed"),
                pl.lit(int(headers["sample_cap"][0])).cast(pl.UInt32).alias("sample_cap"),
                pl.lit(elapsed_seconds).alias("elapsed_seconds"),
                pl.lit(shared_home_bytes).cast(pl.UInt64).alias("index_size_bytes"),
            )
            .pipe(MetricsFile.validate, cast=True)
        )

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(report_text, encoding="utf-8")
        self.metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_json(self.metrics)
