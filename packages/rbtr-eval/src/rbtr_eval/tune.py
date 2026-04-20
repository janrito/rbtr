"""`rbtr-eval tune` subcommand.

Grid-search the rbtr search fusion weights `(alpha, beta,
gamma)` against every per-repo query set, using the
full-variant index in the shared home.  Reports best vs
current weights in `data/tuned-params.json`; never edits
source.

Indexing is a separate DVC stage; this command only queries.
One warm daemon serves every grid point for every query.
Aggregation is declarative polars: one
`group_by([alpha, beta, gamma]).agg(mrr)`, pick the top row.
"""

from __future__ import annotations

import time
from pathlib import Path

import dataframely as dy
import polars as pl
import pygit2
from pydantic import BaseModel, Field

from rbtr.config import config as rbtr_config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import ErrorResponse, SearchRequest, SearchResponse
from rbtr.index.models import IndexVariant
from rbtr.index.search import ScoredResult
from rbtr_eval.extract import Query, load_per_repo
from rbtr_eval.rbtr_cli import daemon_session
from rbtr_eval.schemas import TuneReport, WeightedSearchOutcome


def grid_triples(step: float) -> list[tuple[float, float, float]]:
    """Enumerate `(alpha, beta, gamma)` in `[0, 1]^3` summing to 1 at *step*.

    Pure function; exposed for tests.  At step 0.2 yields 21
    points; 0.1 yields 66; 0.5 yields 6.  Values are rounded
    to 6 decimals so equality compares cleanly.
    """
    if not 0.0 < step <= 1.0:
        msg = f"grid_step must be in (0, 1]; got {step}"
        raise ValueError(msg)
    n = round(1.0 / step)
    return [
        (round(i * step, 6), round(j * step, 6), round((n - i - j) * step, 6))
        for i in range(n + 1)
        for j in range(n + 1 - i)
    ]


# ── Typed search ─────────────────────────────────────────────────────────────


def _search(
    client: DaemonClient,
    repo_path: Path,
    query: str,
    weights: tuple[float, float, float] | None,
) -> list[ScoredResult]:
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
    response = client.send(request)
    if isinstance(response, ErrorResponse):
        msg = f"daemon search failed: {response.message}"
        raise SystemExit(msg)
    if not isinstance(response, SearchResponse):
        msg = f"unexpected daemon response: {type(response).__name__}"
        raise SystemExit(msg)
    return response.results


def _rank_for(query: Query, hits: list[ScoredResult]) -> int | None:
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        if c.file_path == query.file_path and c.scope == query.scope and c.name == query.name:
            return i
    return None


# ── Dataset load ─────────────────────────────────────────────────────────────


def _load_dataset(per_repo_dir: Path) -> dict[str, list[Query]]:
    """Read per-repo JSONLs; return queries grouped by slug."""
    files = sorted(per_repo_dir.glob("*.jsonl"))
    if not files:
        msg = f"no JSONL files under {per_repo_dir}"
        raise SystemExit(msg)
    by_slug: dict[str, list[Query]] = {}
    for path in files:
        header, queries = load_per_repo(path)
        by_slug[header.slug] = queries
    return by_slug


# ── Search execution ─────────────────────────────────────────────────────────


def _run_weight_trials(
    client: DaemonClient,
    queries_by_slug: dict[str, list[Query]],
    repos_dir: Path,
    triples: list[tuple[float, float, float]],
) -> dy.DataFrame[WeightedSearchOutcome]:
    """Run one baseline + every grid-triple search for every query.

    Baseline rows carry null weights (rbtr's configured
    defaults are in effect).  Grid rows carry the triple
    supplied to that call.  Each row dict mirrors
    `WeightedSearchOutcome`; validation on the next line
    enforces the shape at runtime.
    """
    rows: list[dict[str, str | int | float | None]] = []
    for slug, queries in queries_by_slug.items():
        repo_path = (repos_dir / slug).resolve()
        for query in queries:
            rank = _rank_for(query, _search(client, repo_path, query.text, None))
            rows.append(
                {
                    "slug": slug,
                    "label": "baseline",
                    "query_file": query.file_path,
                    "query_scope": query.scope,
                    "query_name": query.name,
                    "alpha": None,
                    "beta": None,
                    "gamma": None,
                    "rank": rank,
                }
            )
        for triple in triples:
            for query in queries:
                rank = _rank_for(query, _search(client, repo_path, query.text, triple))
                rows.append(
                    {
                        "slug": slug,
                        "label": "grid",
                        "query_file": query.file_path,
                        "query_scope": query.scope,
                        "query_name": query.name,
                        "alpha": triple[0],
                        "beta": triple[1],
                        "gamma": triple[2],
                        "rank": rank,
                    }
                )
    return pl.DataFrame(rows).pipe(WeightedSearchOutcome.validate, cast=True)


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


def _best_grid_triple(trials: dy.DataFrame[WeightedSearchOutcome]) -> pl.DataFrame:
    """Return a one-row frame with the best `(alpha, beta, gamma)` and its MRR."""
    return (
        trials.filter(pl.col("label") == "grid")
        .group_by(["alpha", "beta", "gamma"])
        .agg(_mrr_expr())
        .sort("mrr", descending=True)
        .head(1)
    )


def _baseline_stats(trials: dy.DataFrame[WeightedSearchOutcome]) -> pl.DataFrame:
    """Return a one-row frame with baseline MRR and the query count."""
    return trials.filter(pl.col("label") == "baseline").select(
        _mrr_expr(),
        pl.len().cast(pl.UInt32).alias("n_queries"),
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def _resolve_rbtr_sha() -> str:
    try:
        repo = pygit2.Repository(".")
        return str(repo.head.target)
    except (pygit2.GitError, KeyError):
        return "unknown"


class TuneCmd(BaseModel):
    """Grid-search rbtr's fusion weights against the query set."""

    per_repo_dir: Path = Field(description="Directory holding per-repo JSONL files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    home: Path = Field(description="Single RBTR_HOME with the `full` index built.")
    grid_step: float = Field(0.2, description="Step size for the (alpha, beta, gamma) grid.")
    output: Path = Field(description="Output path for the tuning suggestion JSON.")

    def cli_cmd(self) -> None:
        rbtr_sha = _resolve_rbtr_sha()
        queries_by_slug = _load_dataset(self.per_repo_dir)
        triples = grid_triples(self.grid_step)
        t0 = time.monotonic()

        with daemon_session(self.home) as client:
            trials = _run_weight_trials(client, queries_by_slug, self.repos_dir, triples)

        grid_best = _best_grid_triple(trials)
        baseline = _baseline_stats(trials)
        if grid_best.is_empty() or baseline.is_empty():
            msg = "no rows produced; dataset empty?"
            raise SystemExit(msg)

        # Assemble the one-row report with literal metadata columns;
        # `TuneReport.write_json` serialises the typed frame.
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
