"""`rbtr-eval tune` subcommand.

Grid-search the rbtr search fusion weights `(alpha, beta,
gamma)` against every per-repo query set, using the
full-variant index in the shared home.  Reports best vs current
weights in `data/tuned-params.json`; never edits source.

Indexing is a separate DVC stage; this command only queries.
One warm daemon serves every grid point for every query.  The
grid: `(alpha, beta, gamma)` triples in `[0, 1]^3` summing to 1
at `--grid-step` resolution.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import polars as pl
import pygit2
from pydantic import BaseModel, Field

from rbtr.config import config as rbtr_config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import ErrorResponse, SearchRequest, SearchResponse
from rbtr.index.models import IndexVariant
from rbtr.index.search import ScoredResult
from rbtr_eval.extract import Query, load_per_repo


class _GridRow(BaseModel, frozen=True):
    """One row per (slug, label, triple, query): the rank that run scored.

    `label` is either `baseline` (uses rbtr's configured
    weights) or `grid` (uses the supplied triple).  For
    `baseline` rows, alpha/beta/gamma are all None.
    """

    slug: str
    label: str
    alpha: float | None
    beta: float | None
    gamma: float | None
    rank: int | None


class _WeightsTriple(BaseModel, frozen=True):
    alpha: float
    beta: float
    gamma: float


class _TuneReport(BaseModel, frozen=True):
    """Full shape of `tuned-params.json`."""

    current: _WeightsTriple
    best: _WeightsTriple
    metric: str
    score_current: float
    score_best: float
    delta: float
    grid_step: float
    n_queries: int
    n_grid_points: int
    rbtr_sha: str
    elapsed_seconds: float


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
    a = beta = gamma = None
    if weights is not None:
        a, beta, gamma = weights
    request = SearchRequest(
        repo=str(repo_path),
        query=query,
        variant=IndexVariant.FULL,
        limit=10,
        alpha=a,
        beta=beta,
        gamma=gamma,
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

        rows: list[_GridRow] = []
        with _daemon(self.home) as client:
            for slug, queries in queries_by_slug.items():
                repo_path = (self.repos_dir / slug).resolve()
                # Baseline: no override, rbtr's configured defaults apply.
                for q in queries:
                    rows.append(
                        _GridRow(
                            slug=slug,
                            label="baseline",
                            alpha=None,
                            beta=None,
                            gamma=None,
                            rank=_rank_for(q, _search(client, repo_path, q.text, None)),
                        )
                    )
                # Grid pass.
                for t in triples:
                    for q in queries:
                        rows.append(
                            _GridRow(
                                slug=slug,
                                label="grid",
                                alpha=t[0],
                                beta=t[1],
                                gamma=t[2],
                                rank=_rank_for(q, _search(client, repo_path, q.text, t)),
                            )
                        )

        # DuckDB over a polars frame: one MRR per (alpha, beta, gamma), plus the baseline.
        frame = pl.DataFrame(
            [r.model_dump() for r in rows],
            schema={
                "slug": pl.String,
                "label": pl.String,
                "alpha": pl.Float64,
                "beta": pl.Float64,
                "gamma": pl.Float64,
                "rank": pl.Int32,
            },
        )
        con = duckdb.connect(":memory:")
        con.register("ranks", frame)
        grid_best = con.execute(
            """
            SELECT
                alpha,
                beta,
                gamma,
                avg(CASE WHEN rank IS NULL THEN 0.0 ELSE 1.0 / rank END) AS mrr
            FROM ranks
            WHERE label = 'grid'
            GROUP BY alpha, beta, gamma
            ORDER BY mrr DESC
            LIMIT 1
            """
        ).pl()
        baseline = con.execute(
            """
            SELECT
                avg(CASE WHEN rank IS NULL THEN 0.0 ELSE 1.0 / rank END) AS mrr,
                count(*) AS n
            FROM ranks
            WHERE label = 'baseline'
            """
        ).pl()

        if grid_best.is_empty() or baseline.is_empty():
            msg = "no rows produced; dataset empty?"
            raise SystemExit(msg)

        best = grid_best.row(0, named=True)
        base = baseline.row(0, named=True)

        report = _TuneReport(
            current=_WeightsTriple(
                alpha=rbtr_config.search_alpha,
                beta=rbtr_config.search_beta,
                gamma=rbtr_config.search_gamma,
            ),
            best=_WeightsTriple(alpha=best["alpha"], beta=best["beta"], gamma=best["gamma"]),
            metric="MRR",
            score_current=base["mrr"],
            score_best=best["mrr"],
            delta=best["mrr"] - base["mrr"],
            grid_step=self.grid_step,
            n_queries=base["n"],
            n_grid_points=len(triples),
            rbtr_sha=rbtr_sha,
            elapsed_seconds=time.monotonic() - t0,
        )

        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
