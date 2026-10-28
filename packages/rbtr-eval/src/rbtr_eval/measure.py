"""`rbtr-eval measure` subcommand.

Reads per-repo parquet query files, runs every query through
the already-built rbtr index home (one shared home across all
repos), and writes:

* `data/BENCHMARKS.md` - human-readable report.
* `data/metrics.json`  - DVC metrics: each arm's figures, over every
  query and per query kind (`MetricsFile`).

Indexing is a separate DVC stage; this command only queries.
Searches go through `DaemonClient` against a daemon bound to
the shared home.  Ranking is declarative polars (explode +
`int_range().over()` + filter + join).

Every query is searched once per expansion *arm* (`none`,
`keywords`, `variants`, `both` — the full `ArmKind` set).
Metrics are partitioned by arm, so `metrics.json` and the
report's ablation table show the effect of each expansion
channel per query kind.
"""

from __future__ import annotations

import json
import time
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any

import dataframely as dy
import minijinja
import polars as pl
from pydantic import BaseModel, Field

from rbtr.cli.output import progress_reporter
from rbtr.config import config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import SearchRequest, SearchResponse
from rbtr.domain.models import ChunkKind, QueryKind
from rbtr.index.classify import classify_query
from rbtr.index.index import Index
from rbtr_eval.agg import Dimension, SearchMetrics, search_metric_aggs
from rbtr_eval.charts import render_vl_to_png
from rbtr_eval.formatting import md_table
from rbtr_eval.queries import load_all_queries
from rbtr_eval.rbtr_cli import daemon_session
from rbtr_eval.shared_schemas import IDENTITY_COLUMNS, ExpansionRow, QueryRow, RepoHeader


class ArmKind(StrEnum):
    """One expansion configuration measured per query.

    `measure` runs every query under each arm so the ablation
    can isolate the effect of each expansion channel.

    `NONE`     — no expansion (raw query only).
    `KEYWORDS` — keyword expansion only (lexical channel).
    `VARIANTS` — variant expansion only (semantic channel).
    `BOTH`     — both channels.
    """

    NONE = "none"
    KEYWORDS = "keywords"
    VARIANTS = "variants"
    BOTH = "both"


class Hit(dy.Schema):
    """One search result the daemon returned for a query.

    Nested as the struct inside `SearchBatch.hits`, one entry per
    `ScoredResult`; the ranking pipeline in `measure` explodes the list
    into per-hit rows and computes `rank` / `top_*` declaratively.
    """

    file_paths = dy.List(dy.String())
    """Every location the hit's content sits at.

    Identical content is one chunk, so a hit reaches the target file
    when that file is one of these.
    """
    scope = dy.String()
    name = dy.String()
    line_start = dy.UInt32()
    line_end = dy.UInt32()
    symbol_kind = dy.Enum(k.value for k in ChunkKind)


class SearchQuery(dy.Schema):
    """Identity and classification of one measured query.

    The search is identified by `arm`, `slug`, the target
    (`IDENTITY_COLUMNS`, under those same names) and `provenance`; `arm`
    is the expansion configuration and `query_kind` the
    `classify_query` shape.  `SearchBatch` and `SearchOutcome` extend
    this with the raw hits and the scored rank, and a hit's own columns
    carry a `hit_` prefix so the two never collide.
    """

    arm = dy.String(primary_key=True)
    slug = dy.String(primary_key=True)
    language = dy.String(primary_key=True)
    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    line_start = dy.UInt32(primary_key=True)
    line_end = dy.UInt32(primary_key=True)
    symbol_kind = dy.Enum((k.value for k in ChunkKind), primary_key=True)
    provenance = dy.String(primary_key=True)
    query_kind = dy.String()
    query_text = dy.String()
    latency_ms = dy.Float64(min=0.0)


class SearchBatch(SearchQuery):
    """Raw output of `_run_searches` before `_score_outcomes` runs.

    `hits` carries the top-10 results from the daemon as a
    list-of-struct; ranking turns this into `SearchOutcome` with
    scalar `rank` / `top_*` columns.
    """

    hits = dy.List(dy.Struct(Hit.columns()))
    expansion_kind = dy.String(nullable=True)
    expansion_n_keywords = dy.UInt8(nullable=True)
    expansion_n_variants = dy.UInt8(nullable=True)


class SearchOutcome(SearchQuery):
    """One scored search: the rank the target landed at and the top-1
    hit for diagnostics.  `rank` / `top_*` are null when the target
    does not appear in the top 10.
    """

    rank = dy.UInt8(nullable=True, min=1, max=10)
    top_file = dy.String(nullable=True)
    top_line = dy.UInt32(nullable=True)
    top_name = dy.String(nullable=True)
    target_truncated = dy.Bool()


class TruncatedTarget(dy.Schema):
    """A chunk whose embedding input was truncated, keyed like a query's target."""

    slug = dy.String(primary_key=True)
    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    line_start = dy.UInt32(primary_key=True)
    line_end = dy.UInt32(primary_key=True)
    symbol_kind = dy.Enum((k.value for k in ChunkKind), primary_key=True)


class LatencyMetrics(SearchMetrics):
    """`SearchMetrics` plus the searches' latency quantiles."""

    search_p50_ms = dy.Float64(min=0.0)
    search_p95_ms = dy.Float64(min=0.0)


class ArmMetrics(LatencyMetrics):
    """The figures over every query searched under one arm."""

    arm = dy.String(primary_key=True)


class Breakdown(LatencyMetrics):
    """The figures over the queries whose `dimension` column holds `value`, under one arm."""

    arm = dy.String(primary_key=True)
    dimension = dy.Enum((d.value for d in Dimension), primary_key=True)
    value = dy.String(primary_key=True)


class TargetShape(SearchMetrics):
    """The figures per target kind and request shape, under one arm."""

    arm = dy.String(primary_key=True)
    symbol_kind = dy.String(primary_key=True)
    query_kind = dy.String(primary_key=True)


class TruncationImpact(dy.Schema):
    """MRR of the found targets, split by whether their embedding was truncated."""

    target_truncated = dy.Bool(primary_key=True)
    n = dy.UInt32(min=0)
    mrr = dy.Float64(nullable=True, min=0.0, max=1.0)


class MetricsFile(BaseModel):
    """`metrics.json`: the figures `dvc metrics` compares between runs.

    `arms` maps each arm to `all` (every query) and to each query kind,
    each mapping a `LatencyMetrics` column to its value.
    """

    seed: int
    queries_per_cell: int
    elapsed_seconds: float
    shas: dict[str, str]
    arms: dict[str, dict[str, dict[str, int | float | None]]]


_TRUNCATED_CHUNKS_SQL = (
    resources.files("rbtr_eval.sql").joinpath("truncated_chunks.sql").read_text()
)


def _truncated_targets(db_path: Path) -> dy.DataFrame[TruncatedTarget]:
    """The chunks whose embedding input was truncated, read from the index at *db_path*.

    Reads each repo's truncated chunks from its project's file; the
    daemon is stopped by this point.  Empty when no index exists.
    """
    if not db_path.exists():
        return TruncatedTarget.create_empty()

    index = Index(db_path)
    frames = []
    for repo in index.repos():
        with index.reader(repo) as cur:
            frame = cur.sql(_TRUNCATED_CHUNKS_SQL, params={"repo_id": repo.repo_id}).pl()
        frames.append(frame.with_columns(pl.lit(repo.repo_path).alias("path")))
    index.close()
    trunc = pl.concat(frames) if frames else pl.DataFrame()

    if trunc.is_empty():
        return TruncatedTarget.create_empty()

    return (
        trunc.with_columns(pl.col("path").str.split("/").list.last().alias("slug"))
        .select("slug", *IDENTITY_COLUMNS)
        .unique()
        .pipe(TruncatedTarget.validate, cast=True)
    )


def _search(
    client: DaemonClient,
    repo_path: Path,
    query: str,
    *,
    keywords: list[str] | None = None,
    variants: list[str] | None = None,
) -> SearchResponse:
    """One search call via the daemon client, against HEAD.

    Naming `HEAD` explicitly pins the measurement to the snapshot
    the queries were extracted from (`rbtr_eval.corpus`); left
    unset, the daemon would prefer an indexed dirty worktree.
    """
    request = SearchRequest(
        repo_path=str(repo_path),
        query=query,
        limit=10,
        ref="HEAD",
        keywords=keywords,
        variants=variants,
    )
    return client.send_or_raise_as(SearchResponse, request)


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
                        "file_path": query["file_path"],
                        "scope": query["scope"],
                        "name": query["name"],
                        "line_start": query["line_start"],
                        "line_end": query["line_end"],
                        "provenance": query["provenance"],
                        "symbol_kind": query["symbol_kind"],
                        "query_kind": query_kind,
                        "query_text": query["text"],
                        "latency_ms": latency_ms,
                        "hits": [
                            {
                                "file_paths": h.file_paths,
                                "scope": h.scope,
                                "name": h.name,
                                "line_start": h.line_start,
                                "line_end": h.line_end,
                                "symbol_kind": h.kind.value,
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
    # Infer from every row: the expansion_* columns are null for
    # unexpanded arms, so any bounded prefix polars would sample can be
    # all-null and mistype them.
    return pl.DataFrame(rows, infer_schema_length=None).pipe(SearchBatch.validate, cast=True)


def _score_outcomes(
    batch: dy.DataFrame[SearchBatch],
    truncated: dy.DataFrame[TruncatedTarget],
) -> dy.DataFrame[SearchOutcome]:
    """Expand raw hits into ranked + top-hit columns.

    Explodes `hits`, numbers rows within each outcome via
    `int_range().over(...)`, picks the rank of the matching
    target (if any), and the top-1 hit for the misses
    appendix.  Left-joins back so queries whose target never
    appeared keep a null rank.  `target_truncated` holds whether the
    target is one of *truncated*.
    """
    outcome_keys = ["arm", "slug", "language", *IDENTITY_COLUMNS, "provenance"]
    exploded = (
        batch.select(
            *outcome_keys,
            pl.col("hits"),
        )
        .explode("hits", empty_as_null=True)
        .with_columns(
            pl.col("hits").struct.unnest().name.prefix("hit_"),
            pl.int_range(1, pl.len() + 1, dtype=pl.UInt8).over(outcome_keys).alias("hit_rank"),
        )
    )

    ranks = (
        exploded.filter(
            # A hit covers every path its content sits at, so the target
            # file counts as found when it is one of them.
            pl.col("hit_file_paths").list.contains(pl.col("file_path"))
            & (pl.col("hit_scope") == pl.col("scope"))
            & (pl.col("hit_name") == pl.col("name"))
            & (pl.col("hit_line_start") == pl.col("line_start"))
            # The span's end and the kind separate two chunks that begin
            # on one line — without them a hit on the sibling scores as
            # a hit on the target.
            & (pl.col("hit_line_end") == pl.col("line_end"))
            & (pl.col("hit_symbol_kind") == pl.col("symbol_kind"))
        )
        .group_by(outcome_keys)
        .agg(pl.col("hit_rank").min().alias("rank"))
    )

    tops = exploded.filter(pl.col("hit_rank") == 1).select(
        *outcome_keys,
        pl.col("hit_file_paths").list.first().alias("top_file"),
        pl.col("hit_line_start").alias("top_line"),
        pl.col("hit_name").alias("top_name"),
    )

    return (
        batch.drop("hits")
        .join(ranks, on=outcome_keys, how="left")
        .join(tops, on=outcome_keys, how="left")
        .join(
            truncated.with_columns(pl.lit(True).alias("target_truncated")),
            on=["slug", *IDENTITY_COLUMNS],
            how="left",
        )
        .with_columns(pl.col("target_truncated").fill_null(False))
        .pipe(SearchOutcome.validate, cast=True)
    )


def _arm_metrics(outcomes: dy.DataFrame[SearchOutcome]) -> dy.DataFrame[ArmMetrics]:
    """Each arm's figures over every query."""
    return (
        outcomes.group_by("arm")
        .agg(
            *search_metric_aggs(),
            pl.col("latency_ms").quantile(0.5).alias("search_p50_ms"),
            pl.col("latency_ms").quantile(0.95).alias("search_p95_ms"),
        )
        .sort("arm")
        .pipe(ArmMetrics.validate, cast=True)
    )


def _breakdowns(outcomes: dy.DataFrame[SearchOutcome]) -> dy.DataFrame[Breakdown]:
    """Each arm's figures per value of every `Dimension`."""
    return (
        pl.concat(
            outcomes.group_by("arm", dimension.value)
            .agg(
                *search_metric_aggs(),
                pl.col("latency_ms").quantile(0.5).alias("search_p50_ms"),
                pl.col("latency_ms").quantile(0.95).alias("search_p95_ms"),
            )
            .select(
                "arm",
                pl.lit(dimension.value).alias("dimension"),
                pl.col(dimension.value).cast(pl.String).alias("value"),
                pl.exclude("arm", dimension.value),
            )
            for dimension in Dimension
        )
        .sort("arm", "dimension", "value")
        .pipe(Breakdown.validate, cast=True)
    )


def _target_shapes(outcomes: dy.DataFrame[SearchOutcome]) -> dy.DataFrame[TargetShape]:
    """Each arm's figures per target kind and request shape."""
    return (
        outcomes.group_by("arm", pl.col("symbol_kind").cast(pl.String), "query_kind")
        .agg(*search_metric_aggs())
        .sort("arm", "symbol_kind", "query_kind")
        .pipe(TargetShape.validate, cast=True)
    )


def _truncation_impact(
    outcomes: dy.DataFrame[SearchOutcome],
) -> dy.DataFrame[TruncationImpact]:
    """MRR of the found targets, for truncated and for whole embeddings.

    A miss has a null rank and is left out of the mean.
    """
    return (
        outcomes.group_by("target_truncated")
        .agg(pl.len().alias("n"), (1.0 / pl.col("rank")).mean().alias("mrr"))
        .sort("target_truncated")
        .pipe(TruncationImpact.validate, cast=True)
    )


def _metrics_file(
    arm_metrics: dy.DataFrame[ArmMetrics],
    breakdowns: dy.DataFrame[Breakdown],
    headers: dy.DataFrame[RepoHeader],
    elapsed_seconds: float,
) -> MetricsFile:
    """The `metrics.json` content: each arm's figures overall and per query kind."""
    arms: dict[str, dict[str, dict[str, int | float | None]]] = {}
    for row in arm_metrics.iter_rows(named=True):
        arms[row.pop("arm")] = {"all": row}
    by_kind = breakdowns.filter(pl.col("dimension") == Dimension.QUERY_KIND).drop("dimension")
    for row in by_kind.iter_rows(named=True):
        arms[row.pop("arm")][row.pop("value")] = row
    return MetricsFile(
        seed=int(headers["seed"][0]),
        queries_per_cell=int(headers["queries_per_cell"][0]),
        elapsed_seconds=elapsed_seconds,
        shas=dict(headers.select("slug", "sha").iter_rows()),
        arms=arms,
    )


def _as_percent(col: str) -> pl.Expr:
    """*col*, a `0..1` fraction, as text like `30.0%`."""
    return ((pl.col(col) * 100).round(1).cast(pl.String) + pl.lit("%")).alias(col)


def _quality_columns() -> list[pl.Expr]:
    """The display columns every retrieval-quality table shows: n, Hit@k, MRR, NDCG@10."""
    return [
        pl.col("n_queries").alias("n"),
        _as_percent("hit_at_1").alias("Hit@1"),
        _as_percent("hit_at_3").alias("Hit@3"),
        _as_percent("hit_at_10").alias("Hit@10"),
        pl.col("mrr").round(3).cast(pl.String).alias("MRR"),
        pl.col("ndcg_at_10").round(3).cast(pl.String).alias("NDCG@10"),
    ]


def _median_rank_column() -> pl.Expr:
    """`median rank` for display; `-` when no query found its target."""
    return (
        pl.when(pl.col("median_rank").is_null())
        .then(pl.lit("-"))
        .otherwise(pl.col("median_rank").cast(pl.Int64).cast(pl.String))
        .alias("median rank")
    )


def _render_report(
    *,
    headers: dy.DataFrame[RepoHeader],
    n_queries: int,
    elapsed_seconds: float,
    arm_metrics: dy.DataFrame[ArmMetrics],
    breakdowns: dy.DataFrame[Breakdown],
    target_shapes: dy.DataFrame[TargetShape],
    truncation: dy.DataFrame[TruncationImpact],
) -> str:
    """Render `BENCHMARKS.md` from the jinja template.

    Every breakdown reports the no-expansion arm (`none`); the ablation
    table compares the arms.  The truncation table appears only when
    some target was truncated.  Cosmetic markdown formatting (column
    alignment, line wrapping) is left to `just lint-md` / CI.
    """
    baseline = breakdowns.filter(pl.col("arm") == ArmKind.NONE)
    not_found = _as_percent("not_found_pct").alias("not found")

    run_table = pl.DataFrame(
        {
            "field": ["seed", "sample target", "total queries", "elapsed"],
            "value": [
                str(int(headers["seed"][0])),
                f"{int(headers['queries_per_cell'][0])} per (repo, language, kind, provenance)",
                str(n_queries),
                f"{round(elapsed_seconds)} s",
            ],
        }
    )
    repos = pl.concat(
        [
            arm_metrics.filter(pl.col("arm") == ArmKind.NONE).select(
                pl.lit("**all repos**").alias("repo"), pl.exclude("arm")
            ),
            baseline.filter(pl.col("dimension") == Dimension.SLUG)
            .sort("value")
            .select(
                pl.format("`{}`", pl.col("value")).alias("repo"),
                pl.exclude("arm", "dimension", "value"),
            ),
        ]
    )
    headline_table = repos.select("repo", *_quality_columns(), _median_rank_column(), not_found)
    latency_table = repos.select(
        "repo",
        (pl.col("search_p50_ms").round(0).cast(pl.Int64).cast(pl.String) + " ms").alias(
            "search P50"
        ),
        (pl.col("search_p95_ms").round(0).cast(pl.Int64).cast(pl.String) + " ms").alias(
            "search P95"
        ),
    )
    kind_table = (
        baseline.filter(pl.col("dimension") == Dimension.SYMBOL_KIND)
        .sort("mrr", "value", descending=[True, False])
        .select(
            pl.format("`{}`", pl.col("value")).alias("symbol_kind"),
            *_quality_columns(),
            not_found,
        )
    )
    target_shape_table = (
        target_shapes.filter(pl.col("arm") == ArmKind.NONE)
        .pivot(
            on="query_kind",
            on_columns=[kind.value for kind in QueryKind],
            index="symbol_kind",
            values="mrr",
        )
        .sort("symbol_kind")
        .select(
            pl.format("`{}`", pl.col("symbol_kind")).alias("symbol_kind"),
            *(pl.col(kind.value).round(3).cast(pl.String).fill_null("-") for kind in QueryKind),
        )
    )
    language_table = (
        baseline.filter(pl.col("dimension") == Dimension.LANGUAGE)
        .sort("value")
        .select(
            pl.format("`{}`", pl.col("value")).alias("language"),
            *_quality_columns(),
            _median_rank_column(),
            not_found,
        )
    )
    provenance_table = (
        baseline.filter(pl.col("dimension") == Dimension.PROVENANCE)
        .sort("value")
        .select(
            pl.format("`{}`", pl.col("value")).alias("provenance"),
            *_quality_columns(),
            _median_rank_column(),
            not_found,
        )
    )
    arm_kind_table = pl.concat(
        [
            arm_metrics.select("arm", pl.lit("**all**").alias("query kind"), *_quality_columns()),
            breakdowns.filter(pl.col("dimension") == Dimension.QUERY_KIND)
            .sort("value", "arm")
            .select(
                "arm",
                pl.format("`{}`", pl.col("value")).alias("query kind"),
                *_quality_columns(),
            ),
        ]
    )
    truncation_table = truncation.select(
        pl.when(pl.col("target_truncated"))
        .then(pl.lit("truncated"))
        .otherwise(pl.lit("full"))
        .alias("embedding"),
        "n",
        _as_percent("mrr").alias("MRR"),
    )

    template = resources.files("rbtr_eval.templates").joinpath("benchmarks.md.j2").read_text()
    return minijinja.Environment().render_str(
        template,
        run_table=md_table(run_table),
        headline_table=md_table(headline_table),
        latency_table=md_table(latency_table),
        kind_table=md_table(kind_table),
        target_shape_table=md_table(target_shape_table),
        language_table=md_table(language_table),
        provenance_table=md_table(provenance_table),
        arm_kind_table=md_table(arm_kind_table),
        truncation_table=md_table(truncation_table) if truncation["target_truncated"].any() else "",
    )


def _charts(breakdowns: dy.DataFrame[Breakdown]) -> dict[str, dict[str, Any]]:
    """The report's Vega-Lite specs, by name, each holding the no-expansion MRR it plots."""
    baseline = breakdowns.filter(pl.col("arm") == ArmKind.NONE)
    data = {
        "mrr_by_repo": (Dimension.SLUG, "slug"),
        "mrr_by_provenance": (Dimension.PROVENANCE, "provenance"),
        "mrr_by_kind": (Dimension.SYMBOL_KIND, "symbol_kind"),
    }
    specs = {}
    for name, (dimension, field) in data.items():
        spec = json.loads(
            resources.files("rbtr_eval.templates").joinpath(f"{name}.vl.json").read_text()
        )
        spec["data"]["values"] = (
            baseline.filter(pl.col("dimension") == dimension)
            .sort("value")
            .select(pl.col("value").alias(field), "mrr")
            .to_dicts()
        )
        specs[name] = spec
    return specs


class MeasureCmd(BaseModel):
    """Replay queries against an already-built shared index home."""

    per_repo_dir: Path = Field(description="Directory holding per-repo parquet files.")
    concept_dir: Path = Field(description="Directory holding concept parquet files.")
    headers_dir: Path = Field(description="Directory holding header parquet files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
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
        with daemon_session() as client:
            batch = _run_searches(client, queries, self.repos_dir)
        elapsed_seconds = time.monotonic() - t0

        outcomes = _score_outcomes(batch, _truncated_targets(config.db_path))
        arm_metrics = _arm_metrics(outcomes)
        breakdowns = _breakdowns(outcomes)

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(
            _render_report(
                headers=headers,
                n_queries=queries.height,
                elapsed_seconds=elapsed_seconds,
                arm_metrics=arm_metrics,
                breakdowns=breakdowns,
                target_shapes=_target_shapes(outcomes),
                truncation=_truncation_impact(outcomes),
            ),
            encoding="utf-8",
        )
        for name, spec in _charts(breakdowns).items():
            render_vl_to_png(spec, self.report.parent / f"{name}.png")
        self.metrics.parent.mkdir(parents=True, exist_ok=True)
        self.metrics.write_text(
            _metrics_file(arm_metrics, breakdowns, headers, elapsed_seconds).model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )
