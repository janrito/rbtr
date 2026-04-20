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

import json
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]  # no stubs available
import pygit2
from pydantic import BaseModel, Field

from rbtr.config import config as rbtr_config
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


# ── Isolation guard ────────────────────────────────────────────────────────────


def _guard_home(home: Path) -> None:
    real = Path(os.environ.get("RBTR_HOME") or (Path.home() / ".rbtr")).expanduser().resolve()
    requested = home.resolve()
    if requested == real or real.is_relative_to(requested) or requested.is_relative_to(real):
        msg = (
            f"refusing to use --home={requested}: overlaps the user's real "
            f"RBTR_HOME ({real}). Pick a path under data/."
        )
        raise SystemExit(msg)


# ── Subprocess wrappers ────────────────────────────────────────────────────────


def _run_rbtr(
    args: list[str], *, env: dict[str, str], capture: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - trusted args
        ["rbtr", *args],  # noqa: S607 - rbtr on PATH
        env=env,
        check=True,
        capture_output=capture,
        text=capture,
    )


@contextmanager
def _daemon(home: Path) -> Iterator[dict[str, str]]:
    """Start one daemon for *home*; stop it on exit."""
    env = os.environ.copy()
    env["RBTR_HOME"] = str(home)
    home.mkdir(parents=True, exist_ok=True)
    _run_rbtr(["daemon", "start"], env=env)
    try:
        yield env
    finally:
        subprocess.run(
            ["rbtr", "daemon", "stop"],  # noqa: S607
            env=env,
            check=False,
            capture_output=True,
        )


def _search(
    env: dict[str, str],
    repo_path: Path,
    query: str,
    weights: tuple[float, float, float] | None,
) -> list[ScoredResult]:
    """One search against the daemon; None *weights* uses config defaults."""
    args = [
        "--json",
        "search",
        query,
        "--variant",
        "full",
        "--limit",
        "10",
        "--repo-path",
        str(repo_path),
    ]
    if weights is not None:
        a, b, g = weights
        args.extend(["--alpha", str(a), "--beta", str(b), "--gamma", str(g)])
    proc = _run_rbtr(args, env=env, capture=True)
    return [
        ScoredResult.model_validate_json(line) for line in proc.stdout.splitlines() if line.strip()
    ]


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


# ── MRR ────────────────────────────────────────────────────────────────────────


def _mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank, treating None as 0."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


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
        _guard_home(self.home)
        _ = IndexVariant.FULL  # liveness check on the rbtr type surface

        rbtr_sha = _resolve_rbtr_sha()
        queries_by_slug = _load_dataset(self.per_repo_dir)
        triples = grid_triples(self.grid_step)
        t0 = time.monotonic()

        rows: list[dict[str, object]] = []
        with _daemon(self.home) as env:
            for slug, queries in queries_by_slug.items():
                repo_path = (self.repos_dir / slug).resolve()
                # Baseline: no override, rbtr's configured defaults apply.
                for q in queries:
                    rows.append(
                        {
                            "slug": slug,
                            "label": "baseline",
                            "alpha": None,
                            "beta": None,
                            "gamma": None,
                            "rank": _rank_for(q, _search(env, repo_path, q.text, None)),
                        }
                    )
                # Grid pass.
                for t in triples:
                    for q in queries:
                        rows.append(
                            {
                                "slug": slug,
                                "label": "grid",
                                "alpha": t[0],
                                "beta": t[1],
                                "gamma": t[2],
                                "rank": _rank_for(q, _search(env, repo_path, q.text, t)),
                            }
                        )

        # Aggregate with duckdb: one MRR per (alpha, beta, gamma), plus the baseline.
        schema = pa.schema(
            [
                pa.field("slug", pa.string()),
                pa.field("label", pa.string()),
                pa.field("alpha", pa.float64()),
                pa.field("beta", pa.float64()),
                pa.field("gamma", pa.float64()),
                pa.field("rank", pa.int32()),
            ]
        )
        con = duckdb.connect(":memory:")
        con.register("ranks", pa.Table.from_pylist(rows, schema=schema))
        grid_mrr_rows = con.execute(
            """
            SELECT
                alpha,
                beta,
                gamma,
                avg(CASE WHEN rank IS NULL THEN 0.0 ELSE 1.0 / rank END) AS mrr,
                count(*) AS n
            FROM ranks
            WHERE label = 'grid'
            GROUP BY alpha, beta, gamma
            ORDER BY mrr DESC
            LIMIT 1
            """
        ).fetchone()
        baseline_row = con.execute(
            """
            SELECT
                avg(CASE WHEN rank IS NULL THEN 0.0 ELSE 1.0 / rank END) AS mrr,
                count(*) AS n
            FROM ranks
            WHERE label = 'baseline'
            """
        ).fetchone()

        if grid_mrr_rows is None or baseline_row is None:
            msg = "no rows produced; dataset empty?"
            raise SystemExit(msg)

        best_a, best_b, best_g, best_mrr, _n_grid = grid_mrr_rows
        baseline_mrr, n_queries = baseline_row

        report = {
            "current": {
                "alpha": rbtr_config.search_alpha,
                "beta": rbtr_config.search_beta,
                "gamma": rbtr_config.search_gamma,
            },
            "best": {"alpha": float(best_a), "beta": float(best_b), "gamma": float(best_g)},
            "metric": "MRR",
            "score_current": float(baseline_mrr),
            "score_best": float(best_mrr),
            "delta": float(best_mrr) - float(baseline_mrr),
            "grid_step": self.grid_step,
            "n_queries": int(n_queries),
            "n_grid_points": len(triples),
            "rbtr_sha": rbtr_sha,
            "elapsed_seconds": time.monotonic() - t0,
        }

        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
