"""`rbtr-eval measure` subcommand.

Reads per-repo JSONL query files, replays every query through
the already-built rbtr index home (one shared home across all
repos and variants), and writes:

* `data/BENCHMARKS.md` — human-readable report.
* `data/metrics.json`  — DVC metrics (polars-written array).

Indexing is a separate DVC stage; this command only queries.
Searches go through `DaemonClient` against a daemon bound to
the shared home.  Aggregation is pure polars: Hit@k, MRR,
median rank, not-found %, and latency quantiles all come from
one `group_by().agg()` call.
"""

from __future__ import annotations

import subprocess
import time
from importlib import resources
from pathlib import Path

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

# Columns produced by the replay loop.  Declared once so both
# the accumulator (dict of lists) and the polars DataFrame call
# reference the same shape.  `latency_ms` lives on the same
# frame so aggregation and quantiles run in one pass.
_REPLAY_SCHEMA: dict[str, pl.DataType] = {
    "slug": pl.String(),
    "variant": pl.String(),
    "query_file": pl.String(),
    "query_scope": pl.String(),
    "query_name": pl.String(),
    "query_text": pl.String(),
    "rank": pl.Int32(),
    "latency_ms": pl.Float64(),
    "top_file": pl.String(),
    "top_line": pl.Int32(),
    "top_name": pl.String(),
}


# ── Typed search ─────────────────────────────────────────────────────────────────────


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


# ── Dataset load ───────────────────────────────────────────────────────────────


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


# ── Query replay ───────────────────────────────────────────────────────────────


def _rank_for(query: Query, hits: list[ScoredResult]) -> int | None:
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        if c.file_path == query.file_path and c.scope == query.scope and c.name == query.name:
            return i
    return None


def _replay_all(
    client: DaemonClient,
    per_repo_headers: list[Header],
    queries_by_slug: dict[str, list[Query]],
    repos_dir: Path,
) -> pl.DataFrame:
    """Replay every query in every variant into a single polars frame.

    One row per (slug, variant, query) with the rank the query
    scored, the call's latency, and the top-1 hit's location
    for the misses appendix.  Columns match `_REPLAY_SCHEMA`.
    """
    cols: dict[str, list[object]] = {name: [] for name in _REPLAY_SCHEMA}
    for h in per_repo_headers:
        repo_path = (repos_dir / h.slug).resolve()
        queries = queries_by_slug.get(h.slug, [])
        for variant in IndexVariant:
            for q in queries:
                hits, ms = _search(client, repo_path, q.text, variant)
                top = hits[0] if hits else None
                cols["slug"].append(h.slug)
                cols["variant"].append(variant.value)
                cols["query_file"].append(q.file_path)
                cols["query_scope"].append(q.scope)
                cols["query_name"].append(q.name)
                cols["query_text"].append(q.text)
                cols["rank"].append(_rank_for(q, hits))
                cols["latency_ms"].append(ms)
                cols["top_file"].append(top.chunk.file_path if top is not None else None)
                cols["top_line"].append(top.chunk.line_start if top is not None else None)
                cols["top_name"].append(top.chunk.name if top is not None else None)
    return pl.DataFrame(cols, schema=_REPLAY_SCHEMA)


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


# ── Aggregation ────────────────────────────────────────────────────────────────


def _metric_aggs() -> list[pl.Expr]:
    """Polars expressions that compute every headline metric.

    Hit@k counts null ranks as misses (`fill_null(False)`).
    MRR treats null ranks as a reciprocal of 0 (custom
    when/then so `1 / null = null` doesn't flip to null-skip).
    `median_rank` drops nulls before medianing; a variant
    where every query missed returns null, which the renderer
    prints as "-".
    """
    rank = pl.col("rank")
    reciprocal = pl.when(rank.is_null()).then(0.0).otherwise(1.0 / rank.cast(pl.Float64))
    return [
        pl.len().alias("n_queries"),
        (rank <= 1).fill_null(False).mean().alias("hit_at_1"),
        (rank <= 3).fill_null(False).mean().alias("hit_at_3"),
        (rank <= 10).fill_null(False).mean().alias("hit_at_10"),
        reciprocal.mean().alias("mrr"),
        rank.drop_nulls().median().alias("median_rank"),
        rank.is_null().mean().alias("not_found_pct"),
        pl.col("latency_ms").quantile(0.5).alias("search_p50_ms"),
        pl.col("latency_ms").quantile(0.95).alias("search_p95_ms"),
    ]


def _aggregate(replay_df: pl.DataFrame) -> pl.DataFrame:
    """Per-(slug, variant) metrics plus an `__all__` rollup per variant."""
    aggs = _metric_aggs()
    per_pair = replay_df.group_by(["slug", "variant"]).agg(*aggs)
    rollup = (
        replay_df.group_by("variant")
        .agg(*aggs)
        .with_columns(pl.lit("__all__").alias("slug"))
        .select(per_pair.columns)
    )
    return pl.concat([per_pair, rollup], how="vertical").sort(["slug", "variant"])


# ── Notable misses ─────────────────────────────────────────────────────────────


def _select_misses(replay_df: pl.DataFrame, limit: int = 20) -> pl.DataFrame:
    """Top *limit* queries by largest stripped-vs-full rank gap.

    `fill_null(11)` converts "no top-10 hit" into a sentinel
    worse than any real rank, so the subtraction is always
    meaningful.  Queries where stripped did the same or
    better than full are filtered out (`gap > 0`).
    """
    pivoted = replay_df.pivot(
        on="variant",
        index=["slug", "query_file", "query_scope", "query_name", "query_text"],
        values=["rank", "top_file", "top_line", "top_name"],
    )
    return (
        pivoted.with_columns(
            (pl.col("rank_stripped").fill_null(11) - pl.col("rank_full").fill_null(11)).alias("gap")
        )
        .filter(pl.col("gap") > 0)
        .sort(
            ["gap", "slug", "query_file"],
            descending=[True, False, False],
        )
        .head(limit)
    )


# ── Rendering ──────────────────────────────────────────────────────────────────


def _render_report(
    *,
    per_repo_headers: list[Header],
    rbtr_sha: str,
    elapsed_seconds: float,
    metrics_df: pl.DataFrame,
    misses_df: pl.DataFrame,
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


# ── Entry point ────────────────────────────────────────────────────────────────


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
        sha_per_slug = pl.DataFrame(
            {"slug": [h.slug for h in per_repo_headers], "sha": [h.sha for h in per_repo_headers]},
            schema={"slug": pl.String(), "sha": pl.String()},
        )

        t0 = time.monotonic()
        with daemon_session(self.home) as client:
            replay_df = _replay_all(client, per_repo_headers, queries_by_slug, self.repos_dir)
        elapsed_seconds = time.monotonic() - t0
        shared_home_bytes = _index_db_bytes(self.home)

        metrics_df = _aggregate(replay_df)
        misses_df = _select_misses(replay_df)

        report_text = _render_report(
            per_repo_headers=per_repo_headers,
            rbtr_sha=rbtr_sha,
            elapsed_seconds=elapsed_seconds,
            metrics_df=metrics_df,
            misses_df=misses_df,
            shared_home_bytes=shared_home_bytes,
        )

        # Metrics file: metrics frame + run metadata + per-repo
        # SHAs joined in on `slug` (`__all__` rows stay null).
        # `write_json` handles the serialisation; no json.dumps.
        output_df = metrics_df.join(sha_per_slug, on="slug", how="left").with_columns(
            pl.lit(rbtr_sha).alias("rbtr_sha"),
            pl.lit(per_repo_headers[0].seed).alias("seed"),
            pl.lit(per_repo_headers[0].sample_cap).alias("sample_cap"),
            pl.lit(elapsed_seconds).alias("elapsed_seconds"),
            pl.lit(shared_home_bytes).alias("index_size_bytes"),
        )

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(report_text, encoding="utf-8")
        self.metrics.parent.mkdir(parents=True, exist_ok=True)
        output_df.write_json(self.metrics)
