"""`rbtr-eval tune` subcommand.

Grid-search the rbtr search fusion weights `(alpha, beta,
gamma)` against per-repo query labels.  Writes
`data/tuned-params.json` reporting the best triple alongside
the rbtr current per-`QueryKind` defaults.  Never edits source.

Requires `rbtr search --alpha/--beta/--gamma` (P5 product
change).

Subprocess only \u2014 no `rbtr.*` imports.  pygit2 is used
directly for the workspace HEAD SHA (transitive dep).

Architecture mirrors `measure.py`: one rbtr daemon per repo,
sequential across repos to avoid GPU contention during
indexing (D16).  Within one repo, the baseline pass (no
override) and the grid pass (every triple) all hit the warm
daemon so search overhead is bounded by daemon RPC latency,
not Python startup.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pygit2
from pydantic import BaseModel, Field

from rbtr.index.search import ScoredResult
from rbtr_eval.extract import Header, Query, load_per_repo

# ── Subprocess wrappers (duplicated from measure to keep modules independent) ──


def _rbtr_env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["RBTR_HOME"] = str(home)
    return env


def _run_rbtr(
    args: list[str],
    *,
    env: dict[str, str],
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - trusted args
        ["rbtr", *args],  # noqa: S607 - rbtr on PATH (uv-managed)
        env=env,
        check=True,
        capture_output=capture,
        text=capture,
    )


@contextmanager
def daemon_session(home: Path) -> Iterator[None]:
    env = _rbtr_env(home)
    home.mkdir(parents=True, exist_ok=True)
    _run_rbtr(["daemon", "start"], env=env)
    try:
        yield
    finally:
        subprocess.run(
            ["rbtr", "daemon", "stop"],  # noqa: S607 - rbtr on PATH
            env=env,
            check=False,
            capture_output=True,
        )


def _wait_for_index(home: Path, repo_path: Path, *, poll_seconds: float = 1.0) -> None:
    args = ["--json", "status", "--repo-path", str(repo_path)]
    while True:
        proc = _run_rbtr(args, env=_rbtr_env(home), capture=True)
        body = proc.stdout.strip().splitlines()[-1]
        report = json.loads(body)
        if report.get("active_job") is None and not report.get("pending"):
            return
        time.sleep(poll_seconds)


def rbtr_index(home: Path, repo_path: Path) -> None:
    """Build / update the index for *repo_path* under *home* (default mode).

    Tune is orthogonal to docstring stripping (D11), so always
    indexes in default mode.
    """
    args = ["index", "--repo-path", str(repo_path)]
    _run_rbtr(args, env=_rbtr_env(home))
    _wait_for_index(home, repo_path)


def rbtr_search(
    home: Path,
    repo_path: Path,
    query: str,
    *,
    weights: tuple[float, float, float] | None,
    limit: int = 10,
) -> list[ScoredResult]:
    args = [
        "--json",
        "search",
        query,
        "--limit",
        str(limit),
        "--repo-path",
        str(repo_path),
    ]
    if weights is not None:
        a, b, g = weights
        args.extend(["--alpha", str(a), "--beta", str(b), "--gamma", str(g)])
    proc = _run_rbtr(args, env=_rbtr_env(home), capture=True)
    hits: list[ScoredResult] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        hits.append(ScoredResult.model_validate_json(line))
    return hits


# ── Match logic (duplicated from measure, on purpose) ──────────────────────────


def _rank_for(query: Query, hits: list[ScoredResult]) -> int | None:
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        if c.file_path == query.file_path and c.scope == query.scope and c.name == query.name:
            return i
    return None


# ── Grid generation (pure projection; testable per D13) ────────────────────────


def grid_triples(step: float) -> list[tuple[float, float, float]]:
    """Enumerate `(alpha, beta, gamma)` in `[0, 1]^3` summing to 1 at *step*.

    Pure function; exposed for tests.  At step 0.2 yields 21
    points; at 0.1, 66; at 0.5, 6.  Floats are rounded to a
    reasonable precision so equality compares cleanly.
    """
    if not 0.0 < step <= 1.0:
        msg = f"grid_step must be in (0, 1]; got {step}"
        raise ValueError(msg)
    n = round(1.0 / step)
    triples: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            triples.append((round(i * step, 6), round(j * step, 6), round(k * step, 6)))
    return triples


# ── MRR aggregation ────────────────────────────────────────────────────────────


def _mrr(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


# ── Isolation guard ────────────────────────────────────────────────────────────


def _resolve_real_home() -> Path:
    env = os.environ.get("RBTR_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".rbtr").resolve()


def _guard_homes_dir(homes_dir: Path) -> None:
    real = _resolve_real_home()
    requested = homes_dir.resolve()
    if requested == real or real.is_relative_to(requested) or requested.is_relative_to(real):
        msg = (
            f"refusing to use --homes-dir={requested}: overlaps the user's "
            f"real RBTR_HOME ({real}). Pick a path under data/."
        )
        raise SystemExit(msg)


# ── Loader (re-uses extract.load_per_repo) ─────────────────────────────────────


def _load_dataset(per_repo_dir: Path) -> tuple[list[Header], dict[str, list[Query]]]:
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
    return headers, by_slug


def _resolve_rbtr_sha() -> str:
    try:
        repo = pygit2.Repository(".")
        return str(repo.head.target)
    except (pygit2.GitError, KeyError):
        return "unknown"


# ── Current rbtr per-kind weights ──────────────────────────────────────────────
#
# Mirror of `_KIND_WEIGHTS` in `packages/rbtr/src/rbtr/index/search.py`.
# Reported informationally so the operator can compare the tuned
# uniform triple against the per-kind defaults it would replace.
# Drifts from rbtr's source if rbtr's defaults change without
# updating this; not a correctness concern for the tune itself
# because the baseline `score_current` is *measured* (no override).

_RBTR_CURRENT_WEIGHTS = {
    "identifier": {"alpha": 0.1, "beta": 0.0, "gamma": 0.9},
    "concept": {"alpha": 0.4, "beta": 0.1, "gamma": 0.5},
    "pattern": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
}


# ── CLI subcommand ─────────────────────────────────────────────────────────────


class TuneCmd(BaseModel):
    """Grid-search rbtr's search fusion weights against the dataset."""

    per_repo_dir: Path = Field(description="Directory holding per-repo JSONL files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    homes_dir: Path = Field(description="Root for per-repo RBTR_HOME directories.")
    grid_step: float = Field(0.2, description="Step size for the (alpha, beta, gamma) grid.")
    output: Path = Field(description="Output path for the tuning suggestion JSON.")

    def cli_cmd(self) -> None:
        _guard_homes_dir(self.homes_dir)
        rbtr_sha = _resolve_rbtr_sha()
        per_repo_headers, queries_by_slug = _load_dataset(self.per_repo_dir)
        triples = grid_triples(self.grid_step)

        # Per-(triple, query) ranks aggregated across repos.
        # Index `triples` includes the (0, 0, 0) origin if step
        # divides 1 cleanly; the validator on rbtr SearchRequest
        # rejects sums != 1, so the grid generator already only
        # emits triples summing to 1.
        all_ranks_per_triple: dict[tuple[float, float, float], list[int | None]] = {
            t: [] for t in triples
        }
        baseline_ranks: list[int | None] = []

        # Sequential daemon per repo (D16).
        for h in per_repo_headers:
            slug = h.slug
            queries = queries_by_slug.get(slug, [])
            repo_path = (self.repos_dir / slug).resolve()
            home = self.homes_dir / slug / "default"
            with daemon_session(home):
                rbtr_index(home, repo_path)
                # Baseline: no override (rbtr's actual per-kind weights apply).
                for q in queries:
                    hits = rbtr_search(home, repo_path, q.text, weights=None)
                    baseline_ranks.append(_rank_for(q, hits))
                # Grid pass.
                for t in triples:
                    for q in queries:
                        hits = rbtr_search(home, repo_path, q.text, weights=t)
                        all_ranks_per_triple[t].append(_rank_for(q, hits))

        baseline_mrr = _mrr(baseline_ranks)
        scored = [(t, _mrr(ranks)) for t, ranks in all_ranks_per_triple.items()]
        best_triple, best_mrr = max(scored, key=lambda pair: pair[1])
        a, b, g = best_triple

        report = {
            "current": _RBTR_CURRENT_WEIGHTS,
            "best": {"alpha": a, "beta": b, "gamma": g},
            "metric": "MRR",
            "score_current": baseline_mrr,
            "score_best": best_mrr,
            "delta": best_mrr - baseline_mrr,
            "grid_step": self.grid_step,
            "n_queries": len(baseline_ranks),
            "n_grid_points": len(triples),
            "rbtr_sha": rbtr_sha,
        }

        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
