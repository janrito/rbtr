"""`rbtr-eval measure` subcommand.

Reads per-repo JSONL query files, runs every query through
the already-built rbtr index home (one shared home across all
repos and variants), and writes:

* `data/BENCHMARKS.md` — human-readable report.
* `data/metrics.json`  — DVC metrics (polars-written array).

Indexing is a separate DVC stage; this command only queries.
Searches go through `DaemonClient` against a daemon bound to
the shared home.  Aggregation is declarative polars: Hit@k,
MRR, median rank, not-found %, and latency quantiles all come
from `group_by(...).agg(*aggs)` over the `SearchOutcome`
frame.
"""

from __future__ import annotations

import subprocess
import time
from importlib import resources
from pathlib import Path

import dataframely as dy
import minijinja
import polars as pl
import pygit2
from pydantic import BaseModel, Field

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import ErrorResponse, SearchRequest, SearchResponse
from rbtr.index.models import IndexVariant
from rbtr.index.search import ScoredResult
from rbtr_eval.extract import Header, Query, load_per_repo
from rbtr_eval.rbtr_cli import daemon_session
from rbtr_eval.schemas import Metrics, MetricsFile, MissCandidate, SearchOutcome

# ── Typed search ─────────────────────────────────────────────────────────────


def _search(
    client: DaemonClient,
    repo_path: Path,
    query: str,
    variant: IndexVariant,
) -> tuple[list[ScoredResult], float]:
    """One search call via the daemon client; returns (hits, wall_ms)."""
    request = SearchRequest(
        repo=str(repo_path),
        query=query,
        variant=variant,
        limit=10,
    )
    t0 = time.monotonic()
    response = client.send(request)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    if isinstance(response, ErrorResponse):
        msg = f"daemon search failed: {response.message}"
        raise SystemExit(msg)
    if not isinstance(response, SearchResponse):
        msg = f"unexpected daemon response: {type(response).__name__}"
        raise SystemExit(msg)
    return response.results, elapsed_ms


def _rank_for(query: Query, hits: list[ScoredResult]) -> int | None:
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        if c.file_path == query.file_path and c.scope == query.scope and c.name == query.name:
            return i
    return None


# ── Dataset load ─────────────────────────────────────────────────────────────


def _load_dataset(per_repo_dir: Path) -> tuple[list[Header], dict[str, list[Query]]]:
    """Read every `<slug>.jsonl` under *per_repo_dir*.

    Returns the per-repo headers (alphabetical by slug) and
    queries grouped by slug.  Refuses to run if any header
    disagrees on seed / sample_cap.
    """
    files = sorted(per_repo_dir.glob("*.jsonl"))
    if not files:
        msg = f"no JSONL files under {per_repo_dir}"
        raise SystemExit(msg)
    headers: list[Header] = []
    by_slug: dict[str, list[Query]] = {}
    for path in files:
        header, queries = load_per_repo(path)
        headers.append(header)
        by_slug[header.slug] = queries
    seeds = {h.seed for h in headers}
    caps = {h.sample_cap for h in headers}
    if len(seeds) != 1 or len(caps) != 1:
        msg = (
            "per-repo headers disagree on seed / sample_cap; "
            f"seeds={sorted(seeds)} caps={sorted(caps)}"
        )
        raise SystemExit(msg)
    return headers, by_slug


# ── Search execution ─────────────────────────────────────────────────────────


def _run_searches(
    client: DaemonClient,
    per_repo_headers: list[Header],
    queries_by_slug: dict[str, list[Query]],
    repos_dir: Path,
) -> dy.DataFrame[SearchOutcome]:
    """Run every `(repo, variant, query)` search and capture its outcome.

    Iterates the native Python inputs; the query space is
    `header x variant x query`, the natural shape of a
    nested loop.  Each row dict mirrors `SearchOutcome`;
    validation on the next line enforces the shape at runtime.
    """
    rows: list[dict[str, str | int | float | None]] = []
    for header in per_repo_headers:
        repo_path = (repos_dir / header.slug).resolve()
        for variant in IndexVariant:
            for query in queries_by_slug.get(header.slug, []):
                hits, latency_ms = _search(client, repo_path, query.text, variant)
                top = hits[0] if hits else None
                rows.append(
                    {
                        "slug": header.slug,
                        "variant": variant.value,
                        "query_file": query.file_path,
                        "query_scope": query.scope,
                        "query_name": query.name,
                        "query_text": query.text,
                        "rank": _rank_for(query, hits),
                        "latency_ms": latency_ms,
                        "top_file": top.chunk.file_path if top is not None else None,
                        "top_line": top.chunk.line_start if top is not None else None,
                        "top_name": top.chunk.name if top is not None else None,
                    }
                )
    return pl.DataFrame(rows).pipe(SearchOutcome.validate, cast=True)


def _index_db_bytes(home: Path, db_name: str = "index.duckdb") -> int:
    """Size of the rbtr DuckDB index files under *home*.

    Sums `index.duckdb` + `.wal` + `.tmp`; absent files count
    as zero.  Reported once per run since the on-disk file is
    shared across every (slug, variant).
    """
    total = 0
    for sibling in (db_name, db_name + ".wal", db_name + ".tmp"):
        path = home / sibling
        if path.exists():
            total += path.stat().st_size
    return total


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


# ── Rendering ────────────────────────────────────────────────────────────────


def _render_report(
    *,
    per_repo_headers: list[Header],
    rbtr_sha: str,
    elapsed_seconds: float,
    metrics_df: dy.DataFrame[Metrics],
    misses_df: dy.DataFrame[MissCandidate],
    shared_home_bytes: int,
) -> str:
    """Render `BENCHMARKS.md` via the jinja template + rumdl fmt.

    Each jinja row is a plain dict built from a polars row
    (`iter_rows(named=True)`).  `dict[str, Any]` lives only
    inside the template context — see the AGENTS data-handling
    rule's boundary exception.
    """
    template = resources.files("rbtr_eval.templates").joinpath("benchmarks.md.j2").read_text()

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    def bytes_human(n: int | float) -> str:
        n = int(n)
        if n < 1024:
            return f"{n} B"
        if n < 1024 * 1024:
            return f"{n / 1024:.1f} KiB"
        if n < 1024 * 1024 * 1024:
            return f"{n / (1024 * 1024):.1f} MiB"
        return f"{n / (1024 * 1024 * 1024):.1f} GiB"

    env = minijinja.Environment()
    env.add_filter("pct", pct)
    env.add_filter("bytes_human", bytes_human)

    headline_rows = [
        {
            "slug": "**all repos**" if r["slug"] == "__all__" else f"`{r['slug']}`",
            "variant": r["variant"],
            "n_queries": r["n_queries"],
            "hit_at_1": r["hit_at_1"],
            "hit_at_3": r["hit_at_3"],
            "hit_at_10": r["hit_at_10"],
            "mrr": r["mrr"],
            "median_rank_str": "-" if r["median_rank"] is None else str(int(r["median_rank"])),
            "not_found_pct": r["not_found_pct"],
        }
        for r in metrics_df.iter_rows(named=True)
    ]
    latency_rows = [
        {
            "slug": "**all repos**" if r["slug"] == "__all__" else f"`{r['slug']}`",
            "variant": r["variant"],
            "search_p50_ms": r["search_p50_ms"],
            "search_p95_ms": r["search_p95_ms"],
        }
        for r in metrics_df.iter_rows(named=True)
    ]
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

    total_queries = sum(h.n_sampled for h in per_repo_headers)
    rendered = env.render_str(
        template,
        rbtr_sha=rbtr_sha,
        seed=per_repo_headers[0].seed,
        sample_cap=per_repo_headers[0].sample_cap,
        total_queries=total_queries,
        elapsed_seconds=elapsed_seconds,
        per_repo_headers=[h.model_dump() for h in per_repo_headers],
        headline_rows=headline_rows,
        latency_rows=latency_rows,
        shared_home_bytes=shared_home_bytes,
        misses=misses_ctx,
    )

    # rumdl formats the generated output: aligns tables, wraps long
    # lines, normalises spacing.  Keeps the template simple.
    result = subprocess.run(
        ["rumdl", "fmt", "--stdin-filepath", "BENCHMARKS.md", "-"],  # noqa: S607
        input=rendered,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else rendered


def _build_metrics_file(
    *,
    metrics_df: dy.DataFrame[Metrics],
    per_repo_headers: list[Header],
    rbtr_sha: str,
    elapsed_seconds: float,
    shared_home_bytes: int,
) -> dy.DataFrame[MetricsFile]:
    """Attach per-slug SHAs and run metadata for on-disk serialisation.

    `__all__` rollup rows stay null on `sha` after the left
    join; every other row gets its repo's SHA.  Run metadata
    (seed, sample_cap, elapsed_seconds, index_size_bytes,
    rbtr_sha) repeats as literal columns so the JSON file
    carries everything DVC's metrics parser might want.
    """
    shas = pl.DataFrame(
        {"slug": [h.slug for h in per_repo_headers], "sha": [h.sha for h in per_repo_headers]},
        schema={"slug": pl.String(), "sha": pl.String()},
    )
    return (
        metrics_df.join(shas, on="slug", how="left")
        .with_columns(
            pl.lit(rbtr_sha).alias("rbtr_sha"),
            pl.lit(per_repo_headers[0].seed).cast(pl.UInt32).alias("seed"),
            pl.lit(per_repo_headers[0].sample_cap).cast(pl.UInt32).alias("sample_cap"),
            pl.lit(elapsed_seconds).alias("elapsed_seconds"),
            pl.lit(shared_home_bytes).cast(pl.UInt64).alias("index_size_bytes"),
        )
        .pipe(MetricsFile.validate, cast=True)
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def _resolve_rbtr_sha() -> str:
    try:
        repo = pygit2.Repository(".")
        return str(repo.head.target)
    except (pygit2.GitError, KeyError):
        return "unknown"


class MeasureCmd(BaseModel):
    """Replay queries against an already-built shared index home."""

    per_repo_dir: Path = Field(description="Directory holding per-repo JSONL files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    home: Path = Field(description="Single RBTR_HOME holding both index variants.")
    report: Path = Field(description="Output path for BENCHMARKS.md.")
    metrics: Path = Field(description="Output path for metrics JSON.")

    def cli_cmd(self) -> None:
        rbtr_sha = _resolve_rbtr_sha()
        per_repo_headers, queries_by_slug = _load_dataset(self.per_repo_dir)

        t0 = time.monotonic()
        with daemon_session(self.home) as client:
            outcomes = _run_searches(client, per_repo_headers, queries_by_slug, self.repos_dir)
        elapsed_seconds = time.monotonic() - t0
        shared_home_bytes = _index_db_bytes(self.home)

        metrics_df = _aggregate(outcomes)
        misses_df = _select_misses(outcomes)

        report_text = _render_report(
            per_repo_headers=per_repo_headers,
            rbtr_sha=rbtr_sha,
            elapsed_seconds=elapsed_seconds,
            metrics_df=metrics_df,
            misses_df=misses_df,
            shared_home_bytes=shared_home_bytes,
        )
        metrics_file = _build_metrics_file(
            metrics_df=metrics_df,
            per_repo_headers=per_repo_headers,
            rbtr_sha=rbtr_sha,
            elapsed_seconds=elapsed_seconds,
            shared_home_bytes=shared_home_bytes,
        )

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(report_text, encoding="utf-8")
        self.metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_json(self.metrics)
