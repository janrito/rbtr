"""`rbtr-eval measure` subcommand.

Reads per-repo JSONL query files, replays every query through the
already-built rbtr index home (one shared home across all repos
and variants), and writes:

* `data/BENCHMARKS.md` - human-readable report.
* `data/metrics.json`  - DVC metrics.

Indexing is a separate DVC stage; this command only *queries*.
rbtr is exercised via its CLI as a subprocess; types (`Chunk`,
`ScoredResult`) come directly from `rbtr.index.*` per the D9
rule.  Aggregation uses DuckDB (already a transitive dep).
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

import duckdb
import minijinja
import pyarrow as pa  # type: ignore[import-untyped]  # no stubs available
import pygit2
from pydantic import BaseModel, Field

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import ErrorResponse, SearchRequest, SearchResponse
from rbtr.index.models import IndexVariant
from rbtr.index.search import ScoredResult
from rbtr_eval.extract import Header, Query, load_per_repo


class _MetricsRow(BaseModel, frozen=True):
    """Per-(repo, variant) metrics; rolls up into the aggregate too."""

    slug: str
    variant: IndexVariant
    n_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_10: float
    mrr: float
    median_rank: int | None
    not_found_pct: float
    index_size_bytes: int
    search_p50_ms: float
    search_p95_ms: float


class _PerQueryRecord(BaseModel, frozen=True):
    """Per-query rank in each variant; drives the misses appendix."""

    slug: str
    file_path: str
    scope: str
    name: str
    text: str
    full_rank: int | None
    stripped_rank: int | None
    full_top_file: str | None
    full_top_line: int | None
    full_top_name: str | None
    stripped_top_file: str | None
    stripped_top_line: int | None
    stripped_top_name: str | None


# ── Daemon lifecycle + typed search ──────────────────────────────────────────


@contextmanager
def _daemon(home: Path) -> Iterator[DaemonClient]:
    """Start one daemon for *home*; yield a client; stop on exit.

    The measure stage runs one daemon for the entire pass:
    every repo, every variant, every query uses the same warm
    process and shares its loaded embedding model.  The client
    is bound explicitly to *home* (`DaemonClient(sock_dir=home)`),
    so it never picks up the caller's `RBTR_HOME`.
    """
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
    home: Path,
    per_repo_headers: list[Header],
    queries_by_slug: dict[str, list[Query]],
    repos_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, list[float]]]:
    """Replay every query in every variant.

    Returns a flat list of dicts (one per (slug, variant, query))
    for DuckDB aggregation, plus a `(slug, variant) -> index_size`
    map and a `(slug, variant) -> [latencies_ms]` map.
    """
    rows: list[dict[str, object]] = []
    sizes: dict[str, float] = {}
    latencies_by_pair: dict[str, list[float]] = {}

    for h in per_repo_headers:
        slug = h.slug
        repo_path = (repos_dir / slug).resolve()
        queries = queries_by_slug.get(slug, [])
        for variant in IndexVariant:
            latencies: list[float] = []
            for q in queries:
                hits, ms = _search(client, repo_path, q.text, variant)
                latencies.append(ms)
                top = hits[0] if hits else None
                rank = _rank_for(q, hits)
                rows.append(
                    {
                        "slug": slug,
                        "variant": variant.value,
                        "query_file": q.file_path,
                        "query_scope": q.scope,
                        "query_name": q.name,
                        "query_text": q.text,
                        "rank": rank,
                        "top_file": top.chunk.file_path if top is not None else None,
                        "top_line": top.chunk.line_start if top is not None else None,
                        "top_name": top.chunk.name if top is not None else None,
                    }
                )
            key = f"{slug}/{variant.value}"
            latencies_by_pair[key] = latencies
            sizes[key] = _index_db_bytes(home)

    return rows, sizes, latencies_by_pair


def _index_db_bytes(home: Path, db_name: str = "index.duckdb") -> int:
    """Size of the DuckDB index files under *home* (sum of .duckdb + .wal + .tmp)."""
    total = 0
    for sibling in (db_name, db_name + ".wal", db_name + ".tmp"):
        path = home / sibling
        if path.exists():
            total += path.stat().st_size
    return total


# ── Aggregation (DuckDB) ───────────────────────────────────────────────────────


_AGG_SQL = """
SELECT
    slug,
    variant,
    count(*) AS n_queries,
    avg(CASE WHEN rank <= 1 THEN 1.0 ELSE 0.0 END) AS hit_at_1,
    avg(CASE WHEN rank <= 3 THEN 1.0 ELSE 0.0 END) AS hit_at_3,
    avg(CASE WHEN rank <= 10 THEN 1.0 ELSE 0.0 END) AS hit_at_10,
    avg(CASE WHEN rank IS NULL THEN 0.0 ELSE 1.0 / rank END) AS mrr,
    median(rank) FILTER (WHERE rank IS NOT NULL) AS median_rank,
    avg(CASE WHEN rank IS NULL THEN 1.0 ELSE 0.0 END) AS not_found_pct
FROM ranks
GROUP BY slug, variant
UNION ALL
SELECT
    '__all__' AS slug,
    variant,
    count(*) AS n_queries,
    avg(CASE WHEN rank <= 1 THEN 1.0 ELSE 0.0 END) AS hit_at_1,
    avg(CASE WHEN rank <= 3 THEN 1.0 ELSE 0.0 END) AS hit_at_3,
    avg(CASE WHEN rank <= 10 THEN 1.0 ELSE 0.0 END) AS hit_at_10,
    avg(CASE WHEN rank IS NULL THEN 0.0 ELSE 1.0 / rank END) AS mrr,
    median(rank) FILTER (WHERE rank IS NOT NULL) AS median_rank,
    avg(CASE WHEN rank IS NULL THEN 1.0 ELSE 0.0 END) AS not_found_pct
FROM ranks
GROUP BY variant
ORDER BY slug, variant
"""


def _rows_to_arrow(rows: list[dict[str, object]]) -> pa.Table:
    """Build the Arrow table DuckDB registers as `ranks`.

    Extracted so the schema is explicit (rank / top_line are
    nullable int32, the rest are strings) rather than inferred
    from a list of dicts.  DuckDB's replacement-scan only
    accepts Arrow / pandas / polars / ndarrays.
    """
    schema = pa.schema(
        [
            pa.field("slug", pa.string()),
            pa.field("variant", pa.string()),
            pa.field("query_file", pa.string()),
            pa.field("query_scope", pa.string()),
            pa.field("query_name", pa.string()),
            pa.field("query_text", pa.string()),
            pa.field("rank", pa.int32()),
            pa.field("top_file", pa.string()),
            pa.field("top_line", pa.int32()),
            pa.field("top_name", pa.string()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _aggregate(
    rows: list[dict[str, object]],
    sizes: dict[str, float],
    latencies_by_pair: dict[str, list[float]],
) -> list[_MetricsRow]:
    """Aggregate per (slug, variant), plus an `__all__` rollup per variant.

    Hit@k, MRR, median rank, and not-found % come from DuckDB.
    Cost metrics (index size, latency percentiles) come from the
    Python-collected maps -- DuckDB would need a `quantile_cont`
    UDF for percentiles, which is more setup than it saves.
    """
    con = duckdb.connect(":memory:")
    con.register("ranks", _rows_to_arrow(rows))
    results = con.execute(_AGG_SQL).fetchall()
    columns = [desc[0] for desc in con.description]

    # Shared home: one on-disk DB holds every variant, so the
    # size is per-file, not per-(slug, variant).  Every row
    # gets the same shared size; the `__all__` rollup does
    # not double-count.
    shared_bytes = next(iter(sizes.values()), 0) if sizes else 0

    metrics: list[_MetricsRow] = []
    for r in results:
        row = dict(zip(columns, r, strict=True))
        slug = str(row["slug"])
        variant = str(row["variant"])
        key = f"{slug}/{variant}"
        if slug == "__all__":
            lat: list[float] = [
                x for k, lst in latencies_by_pair.items() if k.endswith(f"/{variant}") for x in lst
            ]
        else:
            lat = latencies_by_pair.get(key, [])
        metrics.append(
            _MetricsRow(
                slug=slug,
                variant=IndexVariant(variant),
                n_queries=int(row["n_queries"]),
                hit_at_1=float(row["hit_at_1"]),
                hit_at_3=float(row["hit_at_3"]),
                hit_at_10=float(row["hit_at_10"]),
                mrr=float(row["mrr"]),
                median_rank=None if row["median_rank"] is None else int(row["median_rank"]),
                not_found_pct=float(row["not_found_pct"]),
                index_size_bytes=int(shared_bytes),
                search_p50_ms=_percentile(lat, 50),
                search_p95_ms=_percentile(lat, 95),
            )
        )
    return metrics


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(pct / 100.0 * (len(s) - 1))))
    return s[k]


# ── Notable misses ─────────────────────────────────────────────────────────────


_MISSES_SQL = """
WITH pivoted AS (
    SELECT
        slug,
        query_file,
        query_scope,
        query_name,
        query_text,
        max(CASE WHEN variant = 'full' THEN rank END) AS full_rank,
        max(CASE WHEN variant = 'full' THEN top_file END) AS full_top_file,
        max(CASE WHEN variant = 'full' THEN top_line END) AS full_top_line,
        max(CASE WHEN variant = 'full' THEN top_name END) AS full_top_name,
        max(CASE WHEN variant = 'stripped' THEN rank END) AS stripped_rank,
        max(CASE WHEN variant = 'stripped' THEN top_file END) AS stripped_top_file,
        max(CASE WHEN variant = 'stripped' THEN top_line END) AS stripped_top_line,
        max(CASE WHEN variant = 'stripped' THEN top_name END) AS stripped_top_name
    FROM ranks
    GROUP BY slug, query_file, query_scope, query_name, query_text
)
SELECT *
FROM pivoted
WHERE coalesce(stripped_rank, 11) > coalesce(full_rank, 11)
ORDER BY
    coalesce(stripped_rank, 11) - coalesce(full_rank, 11) DESC,
    slug,
    query_file,
    coalesce(full_top_line, 0)
LIMIT ?
"""


def _select_misses(rows: list[dict[str, object]], limit: int = 20) -> list[_PerQueryRecord]:
    """Top *limit* queries by largest stripped-vs-full rank gap."""
    con = duckdb.connect(":memory:")
    con.register("ranks", _rows_to_arrow(rows))
    fetched = con.execute(_MISSES_SQL, [limit]).fetchall()
    cols = [desc[0] for desc in con.description]
    return [
        _PerQueryRecord(
            slug=str(row["slug"]),
            file_path=str(row["query_file"]),
            scope=str(row["query_scope"]),
            name=str(row["query_name"]),
            text=str(row["query_text"]),
            full_rank=None if row["full_rank"] is None else int(row["full_rank"]),
            stripped_rank=(None if row["stripped_rank"] is None else int(row["stripped_rank"])),
            full_top_file=None if row["full_top_file"] is None else str(row["full_top_file"]),
            full_top_line=None if row["full_top_line"] is None else int(row["full_top_line"]),
            full_top_name=None if row["full_top_name"] is None else str(row["full_top_name"]),
            stripped_top_file=(
                None if row["stripped_top_file"] is None else str(row["stripped_top_file"])
            ),
            stripped_top_line=(
                None if row["stripped_top_line"] is None else int(row["stripped_top_line"])
            ),
            stripped_top_name=(
                None if row["stripped_top_name"] is None else str(row["stripped_top_name"])
            ),
        )
        for row in (dict(zip(cols, r, strict=True)) for r in fetched)
    ]


# ── Rendering ──────────────────────────────────────────────────────────────────


def _render_report(
    *,
    per_repo_headers: list[Header],
    rbtr_sha: str,
    elapsed_seconds: float,
    metrics: list[_MetricsRow],
    misses: list[_PerQueryRecord],
) -> str:
    """Render `BENCHMARKS.md` via the jinja template + rumdl fmt.

    The template lives in `templates/benchmarks.md.j2`.  After
    rendering, the result is piped through `rumdl fmt --stdin-filepath`
    to normalise table alignment and line lengths -- saves us
    re-implementing markdown formatting in the template.
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
            "slug": "**all repos**" if m.slug == "__all__" else f"`{m.slug}`",
            "variant": m.variant.value,
            "n_queries": m.n_queries,
            "hit_at_1": m.hit_at_1,
            "hit_at_3": m.hit_at_3,
            "hit_at_10": m.hit_at_10,
            "mrr": m.mrr,
            "median_rank_str": "-" if m.median_rank is None else str(m.median_rank),
            "not_found_pct": m.not_found_pct,
        }
        for m in metrics
    ]
    latency_rows = [
        {
            "slug": "**all repos**" if m.slug == "__all__" else f"`{m.slug}`",
            "variant": m.variant.value,
            "search_p50_ms": m.search_p50_ms,
            "search_p95_ms": m.search_p95_ms,
        }
        for m in metrics
    ]
    # Shared-home size: any `__all__` row has the full sum; fall
    # back to the first per-repo row if `__all__` is missing (should
    # never happen -- _aggregate always emits the rollup).
    shared_home_bytes = next(
        (m.index_size_bytes for m in metrics if m.slug == "__all__"),
        metrics[0].index_size_bytes if metrics else 0,
    )
    misses_ctx = [
        {
            "slug": m.slug,
            "file_path": m.file_path,
            "scope_name": f"{m.scope}.{m.name}" if m.scope else m.name,
            "text": m.text,
            "full_rank_str": "-" if m.full_rank is None else str(m.full_rank),
            "stripped_rank_str": "-" if m.stripped_rank is None else str(m.stripped_rank),
            "full_top_str": (
                f"{m.full_top_file}:{m.full_top_line} {m.full_top_name}"
                if m.full_top_file is not None
                else "-"
            ),
            "stripped_top_str": (
                f"{m.stripped_top_file}:{m.stripped_top_line} {m.stripped_top_name}"
                if m.stripped_top_file is not None
                else "-"
            ),
        }
        for m in misses
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


def _render_metrics(
    *,
    per_repo_headers: list[Header],
    rbtr_sha: str,
    elapsed_seconds: float,
    metrics: list[_MetricsRow],
) -> dict[str, object]:
    """Full-precision metrics.json payload."""
    sha_for = {h.slug: h.sha for h in per_repo_headers}
    per_repo: dict[str, dict[str, object]] = {}
    aggregate: dict[str, dict[str, object]] = {}
    for m in metrics:
        payload = m.model_dump()
        if m.slug == "__all__":
            aggregate[m.variant.value] = payload
        else:
            per_repo.setdefault(m.slug, {"sha": sha_for[m.slug]})[m.variant.value] = payload
    return {
        "run": {
            "rbtr_sha": rbtr_sha,
            "seed": per_repo_headers[0].seed,
            "sample_cap": per_repo_headers[0].sample_cap,
            "elapsed_seconds": elapsed_seconds,
        },
        "per_repo": per_repo,
        "aggregate": aggregate,
    }


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

        t0 = time.monotonic()
        with _daemon(self.home) as client:
            rows, sizes, latencies = _replay_all(
                client, self.home, per_repo_headers, queries_by_slug, self.repos_dir
            )
        elapsed_seconds = time.monotonic() - t0

        metrics = _aggregate(rows, sizes, latencies)
        misses = _select_misses(rows)

        report_text = _render_report(
            per_repo_headers=per_repo_headers,
            rbtr_sha=rbtr_sha,
            elapsed_seconds=elapsed_seconds,
            metrics=metrics,
            misses=misses,
        )
        metrics_obj = _render_metrics(
            per_repo_headers=per_repo_headers,
            rbtr_sha=rbtr_sha,
            elapsed_seconds=elapsed_seconds,
            metrics=metrics,
        )

        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(report_text, encoding="utf-8")
        self.metrics.parent.mkdir(parents=True, exist_ok=True)
        self.metrics.write_text(json.dumps(metrics_obj, indent=2) + "\n", encoding="utf-8")
