"""`rbtr-eval measure` subcommand.

Reads `data/dataset.jsonl`, builds two `rbtr` indexes per repo
(default + `--strip-docstrings`) into isolated `RBTR_HOME`
directories under *homes-dir*, replays every query through
`rbtr --json search`, and writes:

* `data/BENCHMARKS.md` - human-readable report.
* `data/metrics.json`  - DVC metrics (Hit@1, Hit@3, Hit@10, MRR
  per repo and aggregate, plus index size / build time / search
  latency).

No imports from `rbtr` - rbtr is exercised purely through its
CLI as a subprocess.  pygit2 is used directly for the workspace
HEAD SHA (it's already a transitive dep through rbtr).  Each
`(repo, mode)` pair runs against its own daemon, started and
stopped sequentially so two embedding-model loads never contend
for the GPU at the same time.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pygit2
from pydantic import BaseModel, Field

from rbtr.index.models import Chunk
from rbtr.index.search import ScoredResult
from rbtr_eval.extract import Header, Query, load_per_repo

# ── Types ──────────────────────────────────────────────────────────────────────


Mode = Literal["default", "stripped"]
_MODES: tuple[Mode, ...] = ("default", "stripped")


class _IndexResult(BaseModel, frozen=True):
    elapsed_seconds: float
    db_size_bytes: int


class _RepoMetrics(BaseModel, frozen=True):
    n_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_10: float
    mrr: float
    median_rank: int | None
    not_found_pct: float
    index_size_bytes: int
    index_seconds: float
    search_p50_ms: float
    search_p95_ms: float


class _ModeRollup(BaseModel, frozen=True):
    n_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_10: float
    mrr: float
    median_rank: int | None
    not_found_pct: float


class _PerQueryRecord(BaseModel, frozen=True):
    """Per-query rank in each mode, used for the appendix."""

    slug: str
    file_path: str
    scope: str
    name: str
    text: str
    default_rank: int | None
    stripped_rank: int | None
    default_top: Chunk | None
    stripped_top: Chunk | None


# ── Subprocess wrappers ────────────────────────────────────────────────────────


def _index_db_bytes(home: Path, db_name: str = "index.duckdb") -> int:
    """Return the size of the DuckDB index files under *home*.

    Sums the main DB plus its `.wal` and any `.tmp` siblings.
    Pointedly *not* `du -sb home` because `home` also contains
    the downloaded embedding model (hundreds of MiB) which has
    nothing to do with the search index size we're measuring.
    """
    total = 0
    for sibling in (db_name, db_name + ".wal", db_name + ".tmp"):
        path = home / sibling
        if path.exists():
            total += path.stat().st_size
    return total


def _rbtr_env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["RBTR_HOME"] = str(home)
    return env


def _run_rbtr(
    args: list[str], *, env: dict[str, str], capture: bool = False
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
    """Start the rbtr daemon for *home*; stop it on exit."""
    env = _rbtr_env(home)
    home.mkdir(parents=True, exist_ok=True)
    _run_rbtr(["daemon", "start"], env=env)
    try:
        yield
    finally:
        # daemon stop is best-effort; idle daemon may have already exited.
        subprocess.run(
            ["rbtr", "daemon", "stop"],  # noqa: S607 - rbtr on PATH
            env=env,
            check=False,
            capture_output=True,
        )


def _wait_for_index(home: Path, repo_path: Path, *, poll_seconds: float = 1.0) -> None:
    """Block until the daemon's queue + active job for *repo_path* is empty.

    `rbtr index` against the daemon only enqueues the job and returns
    immediately.  The eval needs the index to actually be ready before
    issuing search calls, so we poll `rbtr --json status` and parse
    the embedded `active_job` / `pending` fields.
    """
    args = ["--json", "status", "--repo-path", str(repo_path)]
    while True:
        proc = _run_rbtr(args, env=_rbtr_env(home), capture=True)
        # The status response is a single JSON object on stdout.
        body = proc.stdout.strip().splitlines()[-1]
        report = json.loads(body)
        if report.get("active_job") is None and not report.get("pending"):
            return
        time.sleep(poll_seconds)


def rbtr_index(home: Path, repo_path: Path, *, strip: bool) -> _IndexResult:
    """Build / update the index for *repo_path* under *home*; block until done.

    Assumes the daemon is already running (see `daemon_session`).
    `rbtr index` itself only queues the job; we poll `rbtr status`
    until the daemon's queue drains.
    """
    args = ["index", "--repo-path", str(repo_path)]
    if strip:
        args.append("--strip-docstrings")
    t0 = time.monotonic()
    _run_rbtr(args, env=_rbtr_env(home))
    _wait_for_index(home, repo_path)
    elapsed = time.monotonic() - t0
    return _IndexResult(elapsed_seconds=elapsed, db_size_bytes=_index_db_bytes(home))


def rbtr_search(
    home: Path,
    repo_path: Path,
    query: str,
    *,
    limit: int = 10,
) -> tuple[list[ScoredResult], float]:
    """Run one search via the daemon; return (hits, wall_clock_ms).

    `--no-daemon` prevents the silent inline fallback that loads the
    embedding model in the search subprocess and crashes on macOS
    cleanup (llama.cpp + atexit metal backend).  We rely on
    `daemon_session` having a running daemon; if the daemon is
    unreachable, this raises rather than silently falling back.
    """
    args = [
        "--json",
        "search",
        query,
        "--limit",
        str(limit),
        "--repo-path",
        str(repo_path),
    ]
    t0 = time.monotonic()
    proc = _run_rbtr(args, env=_rbtr_env(home), capture=True)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    hits: list[ScoredResult] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        hits.append(ScoredResult.model_validate_json(line))
    return hits, elapsed_ms


# ── Match logic ────────────────────────────────────────────────────────────────


def _rank_for(query: Query, hits: list[ScoredResult]) -> int | None:
    """Return 1-based rank of *query*'s labelled chunk, or None."""
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        if c.file_path == query.file_path and c.scope == query.scope and c.name == query.name:
            return i
    return None


# ── Aggregation ────────────────────────────────────────────────────────────────


def _aggregate(
    ranks: list[int | None],
    *,
    n: int,
) -> tuple[float, float, float, float, int | None, float]:
    """Compute (hit@1, hit@3, hit@10, mrr, median_rank, not_found_pct).

    `ranks` is a list of 1-based ranks, with `None` for not-found.
    Hit@k counts ranks <= k.  MRR averages 1/rank, treating
    None as 0.  Median rank is the median of the *found* ranks
    only (None when nothing was found).  not_found_pct is the
    fraction with rank None.
    """
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, None, 0.0
    found = [r for r in ranks if r is not None]
    hit1 = sum(1 for r in found if r <= 1) / n
    hit3 = sum(1 for r in found if r <= 3) / n
    hit10 = sum(1 for r in found if r <= 10) / n
    mrr = sum(1.0 / r for r in found) / n
    median = int(statistics.median(found)) if found else None
    not_found = (n - len(found)) / n
    return hit1, hit3, hit10, mrr, median, not_found


# ── Renderer ───────────────────────────────────────────────────────────────────


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _ms(x: float) -> str:
    return f"{x:.0f} ms"


def _bytes_human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MiB"
    return f"{n / (1024 * 1024 * 1024):.1f} GiB"


def _seconds(x: float) -> str:
    return f"{x:.1f} s"


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a markdown table with column widths aligned to content.

    Output is one line per row.  Satisfies `rumdl`'s MD060
    (column alignment) without us hand-padding format strings.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return [fmt(headers), sep, *[fmt(row) for row in rows]]


def _render_report(
    *,
    per_repo_headers: list[Header],
    rbtr_sha: str,
    elapsed_seconds: float,
    per_repo: dict[str, dict[Mode, _RepoMetrics]],
    rollups: dict[Mode, _ModeRollup],
    misses: list[_PerQueryRecord],
) -> str:
    seed = per_repo_headers[0].seed
    sample_cap = per_repo_headers[0].sample_cap
    lines: list[str] = []
    lines.append("# rbtr search-quality benchmark\n")
    lines.append("Hit@k / MRR for docstring-derived queries against rbtr's default")
    lines.append("and `--strip-docstrings` indexes. See `packages/rbtr-eval/README.md`")
    lines.append("for methodology.\n")
    lines.append("## Run\n")
    lines.append("Reproduce: `cd packages/rbtr-eval && uv run dvc repro`.\n")
    total_q = sum(m["default"].n_queries for m in per_repo.values())
    lines.extend(
        _render_table(
            ["field", "value"],
            [
                ["rbtr commit", f"`{rbtr_sha}`"],
                ["seed", str(seed)],
                ["sample cap", str(sample_cap)],
                ["total queries", str(total_q)],
                ["elapsed", f"{elapsed_seconds:.0f} s"],
            ],
        )
    )
    lines.append("")

    lines.append("## Repos\n")
    lines.extend(
        _render_table(
            ["slug", "sha", "n queries"],
            [[f"`{h.slug}`", f"`{h.sha[:12]}`", str(h.n_sampled)] for h in per_repo_headers],
        )
    )
    lines.append("")

    lines.append("## Headline metrics\n")
    metric_headers = [
        "repo",
        "mode",
        "n",
        "Hit@1",
        "Hit@3",
        "Hit@10",
        "MRR",
        "median rank",
        "not found",
    ]
    metric_rows: list[list[str]] = []
    for slug in sorted(per_repo):
        for mode in _MODES:
            m = per_repo[slug][mode]
            metric_rows.append(
                [
                    f"`{slug}`",
                    mode,
                    str(m.n_queries),
                    _pct(m.hit_at_1),
                    _pct(m.hit_at_3),
                    _pct(m.hit_at_10),
                    f"{m.mrr:.3f}",
                    "-" if m.median_rank is None else str(m.median_rank),
                    _pct(m.not_found_pct),
                ]
            )
    for mode in _MODES:
        roll = rollups[mode]
        metric_rows.append(
            [
                "**all repos**",
                mode,
                str(roll.n_queries),
                _pct(roll.hit_at_1),
                _pct(roll.hit_at_3),
                _pct(roll.hit_at_10),
                f"{roll.mrr:.3f}",
                "-" if roll.median_rank is None else str(roll.median_rank),
                _pct(roll.not_found_pct),
            ]
        )
    lines.extend(_render_table(metric_headers, metric_rows))
    lines.append("")

    lines.append("## Cost of docstrings\n")
    cost_headers = ["repo", "mode", "index size", "index time", "search P50", "search P95"]
    cost_rows: list[list[str]] = []
    for slug in sorted(per_repo):
        for mode in _MODES:
            m = per_repo[slug][mode]
            cost_rows.append(
                [
                    f"`{slug}`",
                    mode,
                    _bytes_human(m.index_size_bytes),
                    _seconds(m.index_seconds),
                    _ms(m.search_p50_ms),
                    _ms(m.search_p95_ms),
                ]
            )
    lines.extend(_render_table(cost_headers, cost_rows))
    lines.append("")

    lines.append("## Notable misses\n")
    if not misses:
        lines.append("_No queries differ between modes._\n")
    else:
        lines.append("Queries where the default-mode rank is much better than")
        lines.append("stripped-mode rank. Sorted by (default_rank - stripped_rank)")
        lines.append("descending; rank `-` means no top-10 match.\n")
        for miss in misses:
            d = "-" if miss.default_rank is None else str(miss.default_rank)
            s = "-" if miss.stripped_rank is None else str(miss.stripped_rank)
            dt = (
                f"{miss.default_top.file_path}:{miss.default_top.line_start} "
                f"{miss.default_top.name}"
                if miss.default_top is not None
                else "-"
            )
            st = (
                f"{miss.stripped_top.file_path}:{miss.stripped_top.line_start} "
                f"{miss.stripped_top.name}"
                if miss.stripped_top is not None
                else "-"
            )
            lines.append("```text")
            lines.append(f"{miss.slug} / {miss.file_path} {miss.scope}.{miss.name}".rstrip("."))
            lines.append(f'  query: "{miss.text}"')
            lines.append(f"  default rank:  {d}")
            lines.append(f"  stripped rank: {s}")
            lines.append(f"  default top:   {dt}")
            lines.append(f"  stripped top:  {st}")
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def _render_metrics(
    *,
    per_repo_headers: list[Header],
    rbtr_sha: str,
    elapsed_seconds: float,
    per_repo: dict[str, dict[Mode, _RepoMetrics]],
    rollups: dict[Mode, _ModeRollup],
) -> dict[str, object]:
    sha_for = {h.slug: h.sha for h in per_repo_headers}
    return {
        "run": {
            "rbtr_sha": rbtr_sha,
            "seed": per_repo_headers[0].seed,
            "sample_cap": per_repo_headers[0].sample_cap,
            "elapsed_seconds": elapsed_seconds,
        },
        "per_repo": {
            slug: {
                "sha": sha_for[slug],
                "n_queries": modes["default"].n_queries,
                "default": modes["default"].model_dump(),
                "stripped": modes["stripped"].model_dump(),
            }
            for slug, modes in per_repo.items()
        },
        "aggregate": {mode: rollups[mode].model_dump() for mode in _MODES},
    }


# ── Isolation guard ────────────────────────────────────────────────────────────


def _resolve_real_home() -> Path:
    """Return the user's real RBTR_HOME (without our override)."""
    env = os.environ.get("RBTR_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".rbtr").resolve()


def _guard_homes_dir(homes_dir: Path) -> None:
    """Refuse to run if *homes_dir* would touch the user's real home."""
    real = _resolve_real_home()
    requested = homes_dir.resolve()
    if requested == real or real.is_relative_to(requested) or requested.is_relative_to(real):
        msg = (
            f"refusing to use --homes-dir={requested}: overlaps the user's "
            f"real RBTR_HOME ({real}). Pick a path under data/."
        )
        raise SystemExit(msg)


# ── CLI subcommand ─────────────────────────────────────────────────────────────


def _resolve_rbtr_sha() -> str:
    """Best-effort: read the rbtr workspace's HEAD SHA via pygit2."""
    try:
        repo = pygit2.Repository(".")
        return str(repo.head.target)
    except (pygit2.GitError, KeyError):
        return "unknown"


def _load_dataset(per_repo_dir: Path) -> tuple[list[Header], dict[str, list[Query]]]:
    """Read every `<slug>.jsonl` under *per_repo_dir*.

    Returns the per-repo headers (one per file, sorted by slug)
    and queries grouped by slug.  Refuses to run if any file is
    missing or any header field disagrees with the others on
    the global knobs (`seed`, `sample_cap`).
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


def _measure_one(
    *,
    home: Path,
    repo_path: Path,
    queries: list[Query],
    strip: bool,
) -> tuple[_IndexResult, list[int | None], list[float], list[ScoredResult | None]]:
    """Index then replay; return (index result, ranks, latencies, top hits).

    `top_hits[i]` is the rank-1 result for query *i*, or None if
    the search returned nothing (used for the misses appendix).
    """
    with daemon_session(home):
        index_result = rbtr_index(home, repo_path, strip=strip)
        ranks: list[int | None] = []
        latencies: list[float] = []
        top_hits: list[ScoredResult | None] = []
        for q in queries:
            hits, ms = rbtr_search(home, repo_path, q.text, limit=10)
            ranks.append(_rank_for(q, hits))
            latencies.append(ms)
            top_hits.append(hits[0] if hits else None)
    return index_result, ranks, latencies, top_hits


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(pct / 100.0 * (len(s) - 1))))
    return s[k]


def _select_misses(
    records: list[_PerQueryRecord],
    *,
    limit: int = 20,
) -> list[_PerQueryRecord]:
    """Pick the top *limit* records by largest stripped-vs-default rank gap.

    Treats None (not-found) as rank 11 so a "default rank 1, stripped
    not-found" case sorts above "default rank 1, stripped rank 8".
    """

    def gap(r: _PerQueryRecord) -> int:
        d = r.default_rank if r.default_rank is not None else 11
        s = r.stripped_rank if r.stripped_rank is not None else 11
        return s - d

    sortable = [r for r in records if gap(r) > 0]
    sortable.sort(
        key=lambda r: (
            -gap(r),
            r.slug,
            r.file_path,
            0 if r.default_top is None else r.default_top.line_start,
        ),
    )
    return sortable[:limit]


class MeasureCmd(BaseModel):
    """Build indexes, replay queries, write report + metrics."""

    per_repo_dir: Path = Field(description="Directory holding per-repo JSONL files.")
    repos_dir: Path = Field(description="Directory holding cloned repos.")
    homes_dir: Path = Field(description="Root for per-(repo, mode) RBTR_HOME directories.")
    report: Path = Field(description="Output path for BENCHMARKS.md.")
    metrics: Path = Field(description="Output path for metrics JSON.")

    def cli_cmd(self) -> None:
        _guard_homes_dir(self.homes_dir)
        rbtr_sha = _resolve_rbtr_sha()
        per_repo_headers, queries_by_slug = _load_dataset(self.per_repo_dir)

        per_repo: dict[str, dict[Mode, _RepoMetrics]] = {}
        per_query_records: list[_PerQueryRecord] = []

        t0 = time.monotonic()
        # Sequential: one daemon at a time across (repo, mode) pairs to avoid
        # GPU contention during indexing.
        for h in per_repo_headers:
            slug = h.slug
            queries = queries_by_slug.get(slug, [])
            repo_path = (self.repos_dir / slug).resolve()
            modes_for_repo: dict[Mode, _RepoMetrics] = {}
            ranks_per_mode: dict[Mode, list[int | None]] = {}
            tops_per_mode: dict[Mode, list[ScoredResult | None]] = {}
            for mode in _MODES:
                home = self.homes_dir / slug / mode
                strip = mode == "stripped"
                index_result, ranks, latencies, tops = _measure_one(
                    home=home,
                    repo_path=repo_path,
                    queries=queries,
                    strip=strip,
                )
                ranks_per_mode[mode] = ranks
                tops_per_mode[mode] = tops
                hit1, hit3, hit10, mrr, median, not_found = _aggregate(ranks, n=len(queries))
                modes_for_repo[mode] = _RepoMetrics(
                    n_queries=len(queries),
                    hit_at_1=hit1,
                    hit_at_3=hit3,
                    hit_at_10=hit10,
                    mrr=mrr,
                    median_rank=median,
                    not_found_pct=not_found,
                    index_size_bytes=index_result.db_size_bytes,
                    index_seconds=index_result.elapsed_seconds,
                    search_p50_ms=_percentile(latencies, 50),
                    search_p95_ms=_percentile(latencies, 95),
                )
            per_repo[slug] = modes_for_repo
            for i, q in enumerate(queries):
                default_top = tops_per_mode["default"][i]
                stripped_top = tops_per_mode["stripped"][i]
                per_query_records.append(
                    _PerQueryRecord(
                        slug=slug,
                        file_path=q.file_path,
                        scope=q.scope,
                        name=q.name,
                        text=q.text,
                        default_rank=ranks_per_mode["default"][i],
                        stripped_rank=ranks_per_mode["stripped"][i],
                        default_top=default_top.chunk if default_top is not None else None,
                        stripped_top=stripped_top.chunk if stripped_top is not None else None,
                    )
                )

        elapsed_seconds = time.monotonic() - t0

        # Aggregate rollups across repos per mode.
        rollups: dict[Mode, _ModeRollup] = {}
        for mode in _MODES:
            ranks_all = [
                rec.default_rank if mode == "default" else rec.stripped_rank
                for rec in per_query_records
            ]
            n = len(ranks_all)
            hit1, hit3, hit10, mrr, median, not_found = _aggregate(ranks_all, n=n)
            rollups[mode] = _ModeRollup(
                n_queries=n,
                hit_at_1=hit1,
                hit_at_3=hit3,
                hit_at_10=hit10,
                mrr=mrr,
                median_rank=median,
                not_found_pct=not_found,
            )

        misses = _select_misses(per_query_records)

        report_text = _render_report(
            per_repo_headers=per_repo_headers,
            rbtr_sha=rbtr_sha,
            elapsed_seconds=elapsed_seconds,
            per_repo=per_repo,
            rollups=rollups,
            misses=misses,
        )
        metrics_obj = _render_metrics(
            per_repo_headers=per_repo_headers,
            rbtr_sha=rbtr_sha,
            elapsed_seconds=elapsed_seconds,
            per_repo=per_repo,
            rollups=rollups,
        )

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(report_text, encoding="utf-8")
        self.metrics.parent.mkdir(parents=True, exist_ok=True)
        self.metrics.write_text(json.dumps(metrics_obj, indent=2) + "\n", encoding="utf-8")
