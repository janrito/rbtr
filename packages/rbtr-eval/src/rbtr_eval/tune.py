"""`rbtr-eval tune` subcommand.

Grid-search the rbtr search fusion weights `(alpha, beta,
gamma)` against every per-repo query set, using the
full-variant index in the shared home.  Reports best vs current
weights in `data/tuned-params.json`; never edits source.

Indexing is a separate DVC stage; this command only queries.
One warm daemon serves every grid point for every query.
Aggregation is pure polars: one `group_by([alpha, beta,
gamma]).agg(mrr)`, pick the top row.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pygit2
from pydantic import BaseModel, Field

from rbtr.config import config as rbtr_config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import ErrorResponse, SearchRequest, SearchResponse
from rbtr.index.models import IndexVariant
from rbtr.index.search import ScoredResult
from rbtr_eval.extract import Query, load_per_repo


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


# Columns for the grid replay frame.  `label` = "baseline" uses
# rbtr's configured defaults; `label` = "grid" uses the supplied
# triple.  For baseline rows, alpha/beta/gamma are all null.
_TUNE_SCHEMA: dict[str, pl.DataType] = {
    "slug": pl.String(),
    "label": pl.String(),
    "alpha": pl.Float64(),
    "beta": pl.Float64(),
    "gamma": pl.Float64(),
    "rank": pl.Int32(),
}


# ── Daemon lifecycle + typed search ──────────────────────────────────────────


@contextmanager
def _daemon(home: Path) -> Iterator[DaemonClient]:
    """Start one daemon for *home*; yield a client; stop on exit."""
    home.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 - trusted args
        ["rbtr", "--home", str(home), "daemon", "start"],  # noqa: S607
        check=True,
    )
    try:
        with DaemonClient(sock_dir=home) as client:
            yield client
    finally:
        subprocess.run(  # noqa: S603 - trusted args
            ["rbtr", "--home", str(home), "daemon", "stop"],  # noqa: S607
            check=False,
            capture_output=True,
        )


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


# ── Dataset load ───────────────────────────────────────────────────────────────


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


# ── Entry point ────────────────────────────────────────────────────────────────


def _resolve_rbtr_sha() -> str:
    try:
        repo = pygit2.Repository(".")
        return str(repo.head.target)
    except (pygit2.GitError, KeyError):
        return "unknown"


def _mrr_expr() -> pl.Expr:
    """MRR expression treating null ranks as reciprocal 0."""
    rank = pl.col("rank")
    return (
        pl.when(rank.is_null()).then(0.0).otherwise(1.0 / rank.cast(pl.Float64)).mean().alias("mrr")
    )


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

        cols: dict[str, list[object]] = {name: [] for name in _TUNE_SCHEMA}
        with _daemon(self.home) as client:
            for slug, queries in queries_by_slug.items():
                repo_path = (self.repos_dir / slug).resolve()
                # Baseline: no override, rbtr's configured defaults apply.
                for q in queries:
                    rank = _rank_for(q, _search(client, repo_path, q.text, None))
                    cols["slug"].append(slug)
                    cols["label"].append("baseline")
                    cols["alpha"].append(None)
                    cols["beta"].append(None)
                    cols["gamma"].append(None)
                    cols["rank"].append(rank)
                # Grid pass.
                for t in triples:
                    for q in queries:
                        rank = _rank_for(q, _search(client, repo_path, q.text, t))
                        cols["slug"].append(slug)
                        cols["label"].append("grid")
                        cols["alpha"].append(t[0])
                        cols["beta"].append(t[1])
                        cols["gamma"].append(t[2])
                        cols["rank"].append(rank)

        ranks_df = pl.DataFrame(cols, schema=_TUNE_SCHEMA)
        grid_best = (
            ranks_df.filter(pl.col("label") == "grid")
            .group_by(["alpha", "beta", "gamma"])
            .agg(_mrr_expr())
            .sort("mrr", descending=True)
            .head(1)
        )
        baseline = ranks_df.filter(pl.col("label") == "baseline").select(
            _mrr_expr(),
            pl.len().alias("n_queries"),
        )

        if grid_best.is_empty() or baseline.is_empty():
            msg = "no rows produced; dataset empty?"
            raise SystemExit(msg)

        # Assemble the output as a one-row frame with literal
        # metadata columns; `write_json` serialises it.  DVC's
        # metrics parser reads the resulting JSON array fine.
        report_df = grid_best.rename(
            {"alpha": "best_alpha", "beta": "best_beta", "gamma": "best_gamma", "mrr": "score_best"}
        ).with_columns(
            pl.lit(rbtr_config.search_alpha).alias("current_alpha"),
            pl.lit(rbtr_config.search_beta).alias("current_beta"),
            pl.lit(rbtr_config.search_gamma).alias("current_gamma"),
            pl.lit(baseline["mrr"][0]).alias("score_current"),
            (pl.lit(grid_best["mrr"][0]) - pl.lit(baseline["mrr"][0])).alias("delta"),
            pl.lit("MRR").alias("metric"),
            pl.lit(self.grid_step).alias("grid_step"),
            pl.lit(baseline["n_queries"][0]).alias("n_queries"),
            pl.lit(len(triples)).alias("n_grid_points"),
            pl.lit(rbtr_sha).alias("rbtr_sha"),
            pl.lit(time.monotonic() - t0).alias("elapsed_seconds"),
        )

        self.output.parent.mkdir(parents=True, exist_ok=True)
        report_df.write_json(self.output)
