"""`rbtr-eval measure` subcommand.

Reads per-repo parquet query files, runs every query through
the already-built rbtr index home (one shared home across all
repos), and writes:

* `data/BENCHMARKS.md` - human-readable report.
* `data/metrics.json`  - DVC metrics (polars-written array).

Indexing is a separate DVC stage; this command only queries.
Searches go through `DaemonClient` against a daemon bound to
the shared home.  Ranking is declarative polars (explode +
`int_range().over()` + filter + join); aggregation is one
`group_by(...).agg(*aggs)` call per grouping level.

Every query is searched once per expansion *arm* (`none`,
`keywords`, `variants`, `both` — the full `ArmKind` set).
Metrics are partitioned by arm, so `metrics.json` and the
report's ablation table show the effect of each expansion
channel per query kind.
"""

from __future__ import annotations

import json
import time
from importlib import resources
from pathlib import Path

import dataframely as dy
import duckdb
import minijinja
import polars as pl
from pydantic import BaseModel, Field

from rbtr.cli.output import progress_reporter
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import SearchRequest, SearchResponse
from rbtr.index.classify import classify_query
from rbtr.index.models import QueryKind
from rbtr_eval.agg import search_metric_aggs
from rbtr_eval.charts import render_vl_to_png
from rbtr_eval.formatting import md_table
from rbtr_eval.queries import load_all_queries
from rbtr_eval.rbtr_cli import daemon_session
from rbtr_eval.schemas import (
    ArmKind,
    ExpansionRow,
    Metrics,
    MetricsFile,
    QueryRow,
    RepoHeader,
    SearchBatch,
    SearchOutcome,
)

_ALL = "__all__"

# ── Typed search ─────────────────────────────────────────────────────────────

_TRUNCATED_CHUNKS_SQL = (
    resources.files("rbtr_eval.sql").joinpath("truncated_chunks.sql").read_text()
)


def _annotate_truncation(
    outcomes: dy.DataFrame[SearchOutcome],
    data_dir: Path,
) -> dy.DataFrame[SearchOutcome]:
    """Set `target_truncated` on outcomes whose target chunk was truncated.

    Opens the index DB read-only (daemon is stopped by this point)
    and left-joins truncated chunk coordinates onto outcomes.
    """
    db_path = data_dir / "index.duckdb"
    if not db_path.exists():
        return outcomes

    with duckdb.connect(str(db_path), read_only=True) as con:
        trunc = con.execute(_TRUNCATED_CHUNKS_SQL).pl()

    if trunc.is_empty():
        return outcomes

    trunc = trunc.with_columns(
        pl.col("path").str.split("/").list.last().alias("slug"),
    ).select(
        "slug",
        pl.col("file_path").alias("query_file"),
        pl.col("scope").alias("query_scope"),
        pl.col("name").alias("query_name"),
        pl.col("line_start").cast(pl.UInt32).alias("query_line_start"),
    )

    join_keys = ["slug", "query_file", "query_scope", "query_name", "query_line_start"]
    trunc_marked = trunc.unique().with_columns(
        pl.lit(True).alias("_is_truncated"),
    )
    return (
        outcomes.drop("target_truncated")
        .join(trunc_marked, on=join_keys, how="left")
        .with_columns(
            pl.col("_is_truncated").fill_null(False).alias("target_truncated"),
        )
        .drop("_is_truncated")
        .pipe(SearchOutcome.validate, cast=True)
    )


def _search(
    client: DaemonClient,
    repo_path: Path,
    query: str,
    *,
    keywords: list[str] | None = None,
    variants: list[str] | None = None,
) -> SearchResponse:
    """One search call via the daemon client."""
    request = SearchRequest(
        repo_path=str(repo_path),
        query=query,
        limit=10,
        keywords=keywords,
        variants=variants,
    )
    return client.send_or_raise_as(SearchResponse, request)


# ── Search execution ─────────────────────────────────────────────────────────


def _arm_inputs(
    arm: ArmKind,
    keywords: list[str] | None,
    variants: list[str] | None,
) -> tuple[list[str] | None, list[str] | None]:
    """Select the keyword/variant inputs a given arm passes to search."""
    kw = keywords if keywords else None
    vr = variants if variants else None
    match arm:
        case ArmKind.NONE:
            return None, None
        case ArmKind.KEYWORDS:
            return kw, None
        case ArmKind.VARIANTS:
            return None, vr
        case ArmKind.BOTH:
            return kw, vr


def _run_searches(
    client: DaemonClient,
    queries: dy.DataFrame[QueryRow],
    repos_dir: Path,
) -> dy.DataFrame[SearchBatch]:
    """Run every query under each `ArmKind`; capture hits + latency + expansion.

    Each query is searched once per arm, the arm controlling
    which expansion channels (keywords / variants) are passed
    to the daemon.  Returns an un-scored outcome frame with a
    `hits: list[struct]` column.  `_score_outcomes` expands
    that into typed `SearchOutcome` rows with declarative
    ranking.  Progress is reported via rbtr's shared
    `progress_reporter`; a live rich bar on interactive
    stderr, a no-op under DVC capture.
    """
    rows = []
    total = queries.height * len(ArmKind)
    done = 0
    with progress_reporter("measure") as (on_progress,):
        for query in queries.iter_rows(named=True):
            repo_path = (repos_dir / query["slug"]).resolve()
            query_kind = classify_query(query["text"]).value
            for arm in ArmKind:
                kw, vr = _arm_inputs(arm, query.get("keywords"), query.get("variants"))
                t0 = time.monotonic()
                resp = _search(
                    client,
                    repo_path,
                    query["text"],
                    keywords=kw,
                    variants=vr,
                )
                latency_ms = (time.monotonic() - t0) * 1000.0
                expanded = kw is not None or vr is not None
                rows.append(
                    {
                        "arm": arm.value,
                        "slug": query["slug"],
                        "language": query["language"],
                        "query_file": query["file_path"],
                        "query_scope": query["scope"],
                        "query_name": query["name"],
                        "query_line_start": query["line_start"],
                        "provenance": query["provenance"],
                        "query_kind": query_kind,
                        "query_text": query["text"],
                        "latency_ms": latency_ms,
                        "hits": [
                            {
                                "file_path": h.file_path,
                                "scope": h.scope,
                                "name": h.name,
                                "line_start": h.line_start,
                            }
                            for h in resp.results
                        ],
                        "expansion_kind": query_kind if expanded else None,
                        "expansion_n_keywords": len(kw or []) if expanded else None,
                        "expansion_n_variants": len(vr or []) if expanded else None,
                    }
                )
                done += 1
                on_progress(done, total)
    return pl.DataFrame(rows).pipe(SearchBatch.validate, cast=True)


def _score_outcomes(batch: dy.DataFrame[SearchBatch]) -> dy.DataFrame[SearchOutcome]:
    """Expand raw hits into ranked + top-hit columns.

    Explodes `hits`, numbers rows within each outcome via
    `int_range().over(...)`, picks the rank of the matching
    target (if any), and the top-1 hit for the misses
    appendix.  Left-joins back so queries whose target never
    appeared keep a null rank.
    """
    outcome_keys = [
        "arm",
        "slug",
        "language",
        "query_file",
        "query_scope",
        "query_name",
        "query_line_start",
        "provenance",
    ]
    exploded = (
        batch.select(
            *outcome_keys,
            pl.col("hits"),
        )
        .explode("hits", empty_as_null=True)
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
            & (pl.col("hit_line_start") == pl.col("query_line_start"))
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
        .with_columns(pl.lit(False).alias("target_truncated"))
        .pipe(SearchOutcome.validate, cast=True)
    )


# ── Aggregation ──────────────────────────────────────────────────────────────


def _aggregate(outcomes: dy.DataFrame[SearchOutcome]) -> dy.DataFrame[Metrics]:
    """Aggregate per arm at several levels into one `Metrics` frame.

    Every level is partitioned by `arm` (always a real value).
    Within an arm, dims the level does not span carry the
    `'__all__'` sentinel:

    1. `(slug, language, provenance)` — finest grain.
    2. `(slug, language)`.
    3. `(language)`.
    4. `(provenance)`.
    5. `(query_kind)` — the expansion ablation, per kind.
    6. global — overall per-arm effect.

    Hit@k counts null ranks as misses (`fill_null(False)`).
    MRR treats null ranks as reciprocal 0 via an explicit
    `when / then / otherwise`.
    """
    latency = pl.col("latency_ms")
    latency_aggs = [
        latency.quantile(0.5).alias("search_p50_ms"),
        latency.quantile(0.95).alias("search_p95_ms"),
    ]
    aggs = [*search_metric_aggs(), *latency_aggs]

    key_cols = ["arm", "slug", "language", "provenance", "query_kind"]
    val_cols = [
        "n_queries",
        "hit_at_1",
        "hit_at_3",
        "hit_at_10",
        "mrr",
        "ndcg_at_10",
        "median_rank",
        "not_found_pct",
        "search_p50_ms",
        "search_p95_ms",
    ]

    per_group = (
        outcomes.group_by("arm", "slug", "language", "provenance")
        .agg(*aggs)
        .with_columns(pl.lit(_ALL).alias("query_kind"))
        .select(key_cols + val_cols)
    )

    per_repo_lang = (
        outcomes.group_by("arm", "slug", "language")
        .agg(*aggs)
        .with_columns(
            pl.lit(_ALL).alias("provenance"),
            pl.lit(_ALL).alias("query_kind"),
        )
        .select(key_cols + val_cols)
    )

    per_repo = (
        outcomes.group_by("arm", "slug")
        .agg(*aggs)
        .with_columns(
            pl.lit(_ALL).alias("language"),
            pl.lit(_ALL).alias("provenance"),
            pl.lit(_ALL).alias("query_kind"),
        )
        .select(key_cols + val_cols)
    )

    lang_roll = (
        outcomes.group_by("arm", "language")
        .agg(*aggs)
        .with_columns(
            pl.lit(_ALL).alias("slug"),
            pl.lit(_ALL).alias("provenance"),
            pl.lit(_ALL).alias("query_kind"),
        )
        .select(key_cols + val_cols)
    )

    provenance_roll = (
        outcomes.group_by("arm", "provenance")
        .agg(*aggs)
        .with_columns(
            pl.lit(_ALL).alias("slug"),
            pl.lit(_ALL).alias("language"),
            pl.lit(_ALL).alias("query_kind"),
        )
        .select(key_cols + val_cols)
    )

    kind_roll = (
        outcomes.group_by("arm", "query_kind")
        .agg(*aggs)
        .with_columns(
            pl.lit(_ALL).alias("slug"),
            pl.lit(_ALL).alias("language"),
            pl.lit(_ALL).alias("provenance"),
        )
        .select(key_cols + val_cols)
    )

    global_roll = (
        outcomes.group_by("arm")
        .agg(*aggs)
        .with_columns(
            pl.lit(_ALL).alias("slug"),
            pl.lit(_ALL).alias("language"),
            pl.lit(_ALL).alias("provenance"),
            pl.lit(_ALL).alias("query_kind"),
        )
        .select(key_cols + val_cols)
    )

    return (
        pl.concat(
            [
                per_group,
                per_repo_lang,
                per_repo,
                lang_roll,
                provenance_roll,
                kind_roll,
                global_roll,
            ],
            how="vertical",
        )
        .sort(key_cols)
        .pipe(Metrics.validate, cast=True)
    )


# ── Home size via DuckDB ─────────────────────────────────────────────────────


def _data_size_bytes(data_dir: Path, db_name: str = "index.duckdb") -> int:
    """Bytes occupied by the rbtr DuckDB index at *data_dir*.

    Queries `pragma_database_size()` (the table-function form
    of `PRAGMA database_size`) so the `block_size * total_blocks`
    product happens in SQL and the Python side gets one scalar.
    Safe to call only after the daemon has stopped (DuckDB
    takes a process-level lock).
    """
    db_path = data_dir / db_name
    if not db_path.exists():
        return 0
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute(
            # sql
            "SELECT block_size * total_blocks AS bytes FROM pragma_database_size()"
        ).fetchone()
    finally:
        con.close()
    return int(row[0]) if row else 0


# ── Rendering ────────────────────────────────────────────────────────────────


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
# (Metrics, RepoHeader) into a polars frame whose
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


# The slug/language/provenance breakdown tables predate the
# arm dimension; they report the no-expansion baseline (arm
# `none`, all kinds) so they keep their historical meaning.
# The expansion ablation lives in its own arm x kind table.


def _headline_table(metrics_df: dy.DataFrame[Metrics]) -> str:
    """`Metrics` -> markdown table for the headline-metrics section.

    Filters to the baseline arm with `language == '__all__'` and
    `provenance == '__all__'` so only the per-repo rollups appear.
    """
    return md_table(
        metrics_df.filter(
            (pl.col("arm") == ArmKind.NONE.value)
            & (pl.col("query_kind") == _ALL)
            & (pl.col("language") == "__all__")
            & (pl.col("provenance") == "__all__")
        ).select(
            _repo_display_expr,
            pl.col("n_queries").alias("n"),
            _pct_str("hit_at_1").alias("Hit@1"),
            _pct_str("hit_at_3").alias("Hit@3"),
            _pct_str("hit_at_10").alias("Hit@10"),
            pl.col("mrr").round(3).cast(pl.String).alias("MRR"),
            pl.col("ndcg_at_10").round(3).cast(pl.String).alias("NDCG@10"),
            pl.when(pl.col("median_rank").is_null())
            .then(pl.lit("-"))
            .otherwise(pl.col("median_rank").cast(pl.Int64).cast(pl.String))
            .alias("median rank"),
            _pct_str("not_found_pct").alias("not found"),
        )
    )


def _latency_table(metrics_df: dy.DataFrame[Metrics]) -> str:
    """`Metrics` -> markdown table for the search-latency section."""
    return md_table(
        metrics_df.filter(
            (pl.col("arm") == ArmKind.NONE.value)
            & (pl.col("query_kind") == _ALL)
            & (pl.col("language") == "__all__")
            & (pl.col("provenance") == "__all__")
        ).select(
            _repo_display_expr,
            (pl.col("search_p50_ms").round(0).cast(pl.Int64).cast(pl.String) + pl.lit(" ms")).alias(
                "search P50"
            ),
            (pl.col("search_p95_ms").round(0).cast(pl.Int64).cast(pl.String) + pl.lit(" ms")).alias(
                "search P95"
            ),
        )
    )


def _provenance_table(metrics_df: dy.DataFrame[Metrics]) -> str:
    """`Metrics` -> markdown table for per-provenance breakdown.

    Shows rows where `slug == '__all__'` and `language == '__all__'`
    but `provenance` is a real provenance value.
    """
    return md_table(
        metrics_df.filter(
            (pl.col("arm") == ArmKind.NONE.value)
            & (pl.col("query_kind") == _ALL)
            & (pl.col("slug") == "__all__")
            & (pl.col("language") == "__all__")
            & (pl.col("provenance") != "__all__")
        )
        .sort("provenance")
        .select(
            pl.format("`{}`", pl.col("provenance")).alias("provenance"),
            pl.col("n_queries").alias("n"),
            _pct_str("hit_at_1").alias("Hit@1"),
            _pct_str("hit_at_3").alias("Hit@3"),
            _pct_str("hit_at_10").alias("Hit@10"),
            pl.col("mrr").round(3).cast(pl.String).alias("MRR"),
            pl.col("ndcg_at_10").round(3).cast(pl.String).alias("NDCG@10"),
            pl.when(pl.col("median_rank").is_null())
            .then(pl.lit("-"))
            .otherwise(pl.col("median_rank").cast(pl.Int64).cast(pl.String))
            .alias("median rank"),
            _pct_str("not_found_pct").alias("not found"),
        )
    )


def _language_table(metrics_df: dy.DataFrame[Metrics]) -> str:
    """`Metrics` -> markdown table for per-language breakdown.

    Shows rows where `slug == '__all__'` but `language` is a
    real language id, ordered by MRR desc.
    """
    return md_table(
        metrics_df.filter(
            (pl.col("arm") == ArmKind.NONE.value)
            & (pl.col("query_kind") == _ALL)
            & (pl.col("slug") == "__all__")
            & (pl.col("language") != "__all__")
            & (pl.col("provenance") == "__all__")
        )
        .sort("language")
        .select(
            pl.format("`{}`", pl.col("language")).alias("language"),
            pl.col("n_queries").alias("n"),
            _pct_str("hit_at_1").alias("Hit@1"),
            _pct_str("hit_at_3").alias("Hit@3"),
            _pct_str("hit_at_10").alias("Hit@10"),
            pl.col("mrr").round(3).cast(pl.String).alias("MRR"),
            pl.col("ndcg_at_10").round(3).cast(pl.String).alias("NDCG@10"),
            pl.when(pl.col("median_rank").is_null())
            .then(pl.lit("-"))
            .otherwise(pl.col("median_rank").cast(pl.Int64).cast(pl.String))
            .alias("median rank"),
            _pct_str("not_found_pct").alias("not found"),
        )
    )


def _repos_table(headers: dy.DataFrame[RepoHeader]) -> str:
    """`RepoHeader` -> markdown table for the per-repo summary."""
    return md_table(
        headers.sort("slug").select(
            pl.format("`{}`", pl.col("slug")).alias("slug"),
            pl.format("`{}`", pl.col("sha").str.slice(0, 12)).alias("sha"),
            pl.col("n_documented").alias("symbols"),
            pl.col("n_queries").alias("sampled queries"),
        )
    )


def _truncation_table(outcomes: dy.DataFrame[SearchOutcome]) -> str:
    """`SearchOutcome` -> markdown table for the truncation-impact section."""
    return md_table(
        outcomes.group_by("target_truncated")
        .agg(
            pl.len().alias("n"),
            (1.0 / pl.col("rank")).mean().alias("mrr"),
        )
        .sort("target_truncated")
        .with_columns(
            pl.when(pl.col("target_truncated"))
            .then(pl.lit("truncated"))
            .otherwise(pl.lit("full"))
            .alias("embedding"),
            _pct_str("mrr").alias("MRR"),
        )
        .select("embedding", "n", "MRR")
    )


def _classification_table(batch: dy.DataFrame[SearchBatch]) -> str:
    """Cross-tab of provenance x heuristic `QueryKind`.

    Shows how `classify_query` routes queries from each
    provenance bucket.  Primary diagnostic for the heuristic.
    """
    classified = batch.select(
        "provenance",
        pl.col("query_text")
        .map_elements(lambda q: classify_query(q).value, return_dtype=pl.String)
        .alias("query_kind"),
    )

    cross = (
        classified.group_by("provenance", "query_kind")
        .len()
        .pivot(on="query_kind", index="provenance", values="len")
    )

    # Ensure all three kinds appear as columns.
    for kind in QueryKind:
        if kind.value not in cross.columns:
            cross = cross.with_columns(pl.lit(0).alias(kind.value))

    # Compute percentages per provenance row.
    cross = cross.with_columns(
        pl.sum_horizontal(pl.col(k) for k in QueryKind).alias("total"),
    )
    return md_table(
        cross.sort("provenance").select(
            pl.format("`{}`", pl.col("provenance")).alias("provenance"),
            *(
                pl.format(
                    "{}%",
                    (pl.col(kind) * 100 / pl.col("total")).round(1).cast(pl.String),
                ).alias(kind.name)
                for kind in QueryKind
            ),
            pl.col("total").alias("n"),
        )
    )


def _arm_kind_table(metrics_df: dy.DataFrame[Metrics]) -> str:
    """`Metrics` -> markdown table for the expansion ablation.

    The headline experimental result: MRR + Hit@k for each
    `(arm, query_kind)`, plus the per-arm global row
    (`query_kind == '__all__'`).  Reads the
    `slug == language == provenance == '__all__'` rollup so
    each cell aggregates over all repos.
    """
    rows = metrics_df.filter(
        (pl.col("slug") == _ALL) & (pl.col("language") == _ALL) & (pl.col("provenance") == _ALL)
    )
    return md_table(
        rows.sort("query_kind", "arm").select(
            pl.col("arm").alias("arm"),
            pl.when(pl.col("query_kind") == _ALL)
            .then(pl.lit("**all**"))
            .otherwise(pl.format("`{}`", pl.col("query_kind")))
            .alias("query kind"),
            pl.col("n_queries").alias("n"),
            _pct_str("hit_at_1").alias("Hit@1"),
            _pct_str("hit_at_3").alias("Hit@3"),
            _pct_str("hit_at_10").alias("Hit@10"),
            pl.col("mrr").round(3).cast(pl.String).alias("MRR"),
            pl.col("ndcg_at_10").round(3).cast(pl.String).alias("NDCG@10"),
        )
    )


def _render_report(
    *,
    headers: dy.DataFrame[RepoHeader],
    n_queries: int,
    elapsed_seconds: float,
    metrics_df: dy.DataFrame[Metrics],
    outcomes: dy.DataFrame[SearchOutcome],
    shared_home_bytes: int,
    batch: dy.DataFrame[SearchBatch],
    report_dir: Path | None = None,
) -> str:
    """Render `BENCHMARKS.md` from the jinja template.

    Tables come from polars (`pl.Config(tbl_formatting='MARKDOWN')`)
    pre-rendered as strings.  Cosmetic markdown formatting
    (column alignment, line wrapping) is left to
    `just lint-md` / CI; this stage emits whatever polars and
    jinja produce.
    """
    template = resources.files("rbtr_eval.templates").joinpath("benchmarks.md.j2").read_text()

    run_table = pl.DataFrame(
        {
            "field": [
                "seed",
                "sample target",
                "total queries",
                "elapsed",
            ],
            "value": [
                str(int(headers["seed"][0])),
                f"{int(headers['queries_per_cell'][0])} per (repo, language, provenance)",
                str(n_queries),
                f"{round(elapsed_seconds)} s",
            ],
        }
    )

    # Charts: render to PNG if report_dir is provided.
    if report_dir is not None:
        repo_data = (
            metrics_df.filter(
                (pl.col("arm") == ArmKind.NONE.value)
                & (pl.col("query_kind") == _ALL)
                & (pl.col("language") == "__all__")
                & (pl.col("provenance") == "__all__")
                & (pl.col("slug") != "__all__")
            )
            .select("slug", "mrr")
            .to_dicts()
        )
        repo_spec = json.loads(
            resources.files("rbtr_eval.templates").joinpath("mrr_by_repo.vl.json").read_text()
        )
        repo_spec["data"]["values"] = repo_data
        render_vl_to_png(repo_spec, report_dir / "mrr_by_repo.png")

        provenance_data = (
            metrics_df.filter(
                (pl.col("arm") == ArmKind.NONE.value)
                & (pl.col("query_kind") == _ALL)
                & (pl.col("slug") == "__all__")
                & (pl.col("language") == "__all__")
                & (pl.col("provenance") != "__all__")
            )
            .select("provenance", "mrr")
            .to_dicts()
        )
        provenance_spec = json.loads(
            resources.files("rbtr_eval.templates").joinpath("mrr_by_provenance.vl.json").read_text()
        )
        provenance_spec["data"]["values"] = provenance_data
        render_vl_to_png(provenance_spec, report_dir / "mrr_by_provenance.png")

    has_truncation = outcomes["target_truncated"].any()
    trunc_table = _truncation_table(outcomes) if has_truncation else ""

    return minijinja.Environment().render_str(
        template,
        shared_home_bytes_human=_bytes_human(shared_home_bytes),
        run_table=md_table(run_table),
        repos_table=_repos_table(headers),
        headline_table=_headline_table(metrics_df),
        latency_table=_latency_table(metrics_df),
        classification_table=_classification_table(batch),
        language_table=_language_table(metrics_df),
        provenance_table=_provenance_table(metrics_df),
        arm_kind_table=_arm_kind_table(metrics_df),
        truncation_table=trunc_table,
    )


# ── Entry point ──────────────────────────────────────────────────────────────


class MeasureCmd(BaseModel):
    """Replay queries against an already-built shared index home."""

    per_repo_dir: Path = Field(description="Directory holding per-repo parquet files.")
    concept_dir: Path = Field(description="Directory holding concept parquet files.")
    headers_dir: Path = Field(description="Directory holding header parquet files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    data_dir: Path = Field(description="Directory for the DuckDB index.")
    config_dir: Path = Field(description="Directory for config.")
    log_dir: Path = Field(description="Directory for logs.")
    report: Path = Field(description="Output path for BENCHMARKS.md.")
    metrics: Path = Field(description="Output path for metrics JSON.")
    expansion_dir: Path = Field(
        description="Directory holding expansion parquet. "
        "keywords/variants are joined onto queries before search.",
    )

    def cli_cmd(self) -> None:
        queries = load_all_queries(self.per_repo_dir, self.concept_dir)
        headers = pl.read_parquet(self.headers_dir / "*.parquet").pipe(
            RepoHeader.validate, cast=True
        )

        exp = pl.read_parquet(
            self.expansion_dir / "expansions.parquet",
        ).pipe(ExpansionRow.validate, cast=True)
        # Left-join adds keywords/variants columns; the frame is
        # still a valid QueryRow superset — extra columns are
        # read by _run_searches via row dicts.
        queries = queries.join(  # type: ignore[assignment]  # wider than QueryRow
            exp,
            on=ExpansionRow.primary_key(),
            how="left",
        )

        t0 = time.monotonic()
        with daemon_session(self.data_dir, self.config_dir, self.log_dir) as client:
            batch = _run_searches(client, queries, self.repos_dir)
        elapsed_seconds = time.monotonic() - t0
        shared_home_bytes = _data_size_bytes(self.data_dir)

        outcomes = _score_outcomes(batch)
        outcomes = _annotate_truncation(outcomes, self.data_dir)
        metrics_df = _aggregate(outcomes)

        self.report.parent.mkdir(parents=True, exist_ok=True)
        report_text = _render_report(
            headers=headers,
            n_queries=queries.height,
            elapsed_seconds=elapsed_seconds,
            metrics_df=metrics_df,
            outcomes=outcomes,
            shared_home_bytes=shared_home_bytes,
            batch=batch,
            report_dir=self.report.parent,
        )

        # metrics.json: metrics frame + per-slug SHA (joined) + run
        # metadata as literal columns.  `__all__` rollup rows stay
        # null on sha.
        metrics_file = (
            metrics_df.join(headers.select("slug", "sha"), on="slug", how="left")
            .with_columns(
                pl.lit(int(headers["seed"][0])).cast(pl.UInt32).alias("seed"),
                pl.lit(int(headers["queries_per_cell"][0]))
                .cast(pl.UInt32)
                .alias("queries_per_cell"),
                pl.lit(elapsed_seconds).alias("elapsed_seconds"),
                pl.lit(shared_home_bytes).cast(pl.UInt64).alias("index_size_bytes"),
            )
            .pipe(MetricsFile.validate, cast=True)
        )

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(report_text, encoding="utf-8")
        self.metrics.parent.mkdir(parents=True, exist_ok=True)
        self.metrics.write_text(
            json.dumps(metrics_file.to_dicts(), indent=2) + "\n",
            encoding="utf-8",
        )
