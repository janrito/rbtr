#!/usr/bin/env python3
"""Benchmark indexing performance and query latency.

Measures:
  1. Parse + chunk + DB insert time (no embedding), phased breakdown.
  2. Query latency: get_chunks, search_by_name, search_fulltext.
  3. Peak memory (RSS) at each phase.
  4. DB file size after indexing.
  5. Incremental update time (base → head).
  6. Per-language and per-kind statistics.

Usage::

    just bench                       # current repo, HEAD
    just bench -- /path/to/repo main # custom repo and ref
    just bench -- . main feature     # with incremental update

For line-level memory profiling, run under scalene::

    just bench-scalene -- /path/to/repo main
"""

from __future__ import annotations

import gc
import resource
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch


def _rss_mb() -> float:
    """Current peak RSS in MiB (macOS/Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS returns bytes, Linux returns KiB.
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _fmt_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    return f"{size_bytes / (1024 * 1024):.1f} MiB"


def _section(title: str) -> None:
    print(f"\n{'─' * 4} {title} {'─' * (60 - len(title))}")


# ── Phase-level timing ───────────────────────────────────────────────


def _bench_file_listing(repo, ref: str) -> tuple[list, float, dict[str, int]]:
    """Benchmark file listing and return language distribution."""
    from rbtr_legacy.index.git import list_files
    from rbtr_legacy.plugins.manager import get_manager

    t0 = time.monotonic()
    files = list(list_files(repo, ref))
    elapsed = time.monotonic() - t0

    mgr = get_manager()
    lang_counts: dict[str, int] = Counter()
    for f in files:
        lang = mgr.detect_language(f.path) or "unknown"
        lang_counts[lang] += 1

    return files, elapsed, dict(lang_counts.most_common())


def _bench_extraction(files: list) -> tuple[list, float, dict[str, int]]:
    """Benchmark chunk extraction (parsing + tree-sitter + chunking)."""
    from rbtr_legacy.index.orchestrator import _extract_file

    kind_counts: dict[str, int] = Counter()
    all_chunks = []
    t0 = time.monotonic()
    for entry in files:
        try:
            chunks = _extract_file(entry)
            for c in chunks:
                kind_counts[c.kind] += 1
            all_chunks.extend(chunks)
        except Exception as exc:
            print(f"  ERROR: {entry.path}: {exc}")
    elapsed = time.monotonic() - t0
    return all_chunks, elapsed, dict(kind_counts.most_common())


def _bench_db_insert(store, chunks: list, snapshot_rows: list) -> float:
    """Benchmark DB inserts (chunks + snapshots)."""
    t0 = time.monotonic()
    store.insert_chunks(chunks)
    store.insert_snapshots(snapshot_rows)
    elapsed = time.monotonic() - t0
    return elapsed


def _bench_edge_inference(store, chunks: list, files: list, commit_sha: str) -> tuple[int, float]:
    """Benchmark edge inference."""
    from rbtr_legacy.index.edges import infer_doc_edges, infer_import_edges, infer_test_edges

    repo_files = {entry.path for entry in files}
    t0 = time.monotonic()
    edges = []
    edges.extend(infer_import_edges(chunks, repo_files))
    edges.extend(infer_test_edges(chunks, repo_files))
    edges.extend(infer_doc_edges(chunks))
    store.delete_edges(commit_sha)
    store.insert_edges(edges, commit_sha)
    elapsed = time.monotonic() - t0
    return len(edges), elapsed


def _bench_fts(store) -> float:
    """Benchmark FTS index rebuild."""
    t0 = time.monotonic()
    store.rebuild_fts_index()
    elapsed = time.monotonic() - t0
    return elapsed


# ── Query benchmarks ─────────────────────────────────────────────────


def _bench_queries(store, ref: str, chunk_count: int) -> None:
    """Benchmark query latency with detailed stats."""
    from rbtr_legacy.index.models import ChunkKind

    _section("Query latency")
    runs = 10

    # get_chunks (full scan)
    t0 = time.monotonic()
    for _ in range(runs):
        result = store.get_chunks(ref)
    elapsed = (time.monotonic() - t0) / runs
    print(f"  get_chunks ({len(result)} chunks):   {_fmt_time(elapsed)} avg ({runs} runs)")

    # get_chunks with filter
    t0 = time.monotonic()
    for _ in range(runs):
        store.get_chunks(ref, kind=ChunkKind.FUNCTION)
    elapsed = (time.monotonic() - t0) / runs
    print(f"  get_chunks(kind=FUNCTION):   {_fmt_time(elapsed)} avg ({runs} runs)")

    # search_by_name
    patterns = ["parse", "test_", "handle", "config", "init", "get", "set", "create", "delete"]
    t0 = time.monotonic()
    total_results = 0
    for pat in patterns:
        results = store.search_by_name(ref, pat)
        total_results += len(results)
    elapsed = (time.monotonic() - t0) / len(patterns)
    print(f"  search_by_name:              {_fmt_time(elapsed)} avg ({total_results} total hits)")

    # search_fulltext
    queries = [
        "error handling",
        "database connection",
        "HTTP request",
        "parse config",
        "test setup",
        "authentication",
        "file path",
        "import module",
    ]
    t0 = time.monotonic()
    total_results = 0
    for q in queries:
        results = store.search_fulltext(ref, q)
        total_results += len(results)
    elapsed = (time.monotonic() - t0) / len(queries)
    print(f"  search_fulltext (BM25):      {_fmt_time(elapsed)} avg ({total_results} total hits)")

    # get_edges
    t0 = time.monotonic()
    for _ in range(runs):
        edges = store.get_edges(ref)
    elapsed = (time.monotonic() - t0) / runs
    print(f"  get_edges ({len(edges)} edges):    {_fmt_time(elapsed)} avg ({runs} runs)")

    # count_orphan_chunks
    t0 = time.monotonic()
    for _ in range(runs):
        store.count_orphan_chunks()
    elapsed = (time.monotonic() - t0) / runs
    print(f"  count_orphan_chunks:         {_fmt_time(elapsed)} avg ({runs} runs)")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    import pygit2

    from rbtr_legacy.index.orchestrator import build_index, update_index
    from rbtr_legacy.index.store import IndexStore

    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    base_ref = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    head_ref = sys.argv[3] if len(sys.argv) > 3 else None

    repo = pygit2.Repository(repo_path)
    resolved_path = Path(repo_path).resolve()
    print(f"repo:     {resolved_path}")
    print(f"base ref: {base_ref}")
    if head_ref:
        print(f"head ref: {head_ref}")
    print(f"pid:      {sys.executable} (PID {__import__('os').getpid()})")

    rss_start = _rss_mb()
    print(f"RSS at start: {rss_start:.0f} MiB")

    # ── Phase 1: File listing ────────────────────────────────────
    _section("File listing + language detection")
    files, list_time, lang_dist = _bench_file_listing(repo, base_ref)
    rss_after_list = _rss_mb()
    print(f"  files:     {len(files)}")
    print(f"  time:      {_fmt_time(list_time)}")
    print(f"  peak RSS:  {rss_after_list:.0f} MiB (+{rss_after_list - rss_start:.0f})")
    print("  languages:")
    for lang, count in lang_dist.items():
        print(f"    {lang:20s} {count:>6d} files")

    # ── Phase 2: Extraction ──────────────────────────────────────
    _section("Chunk extraction (tree-sitter + plaintext)")
    all_chunks, extract_time, kind_dist = _bench_extraction(files)
    rss_after_extract = _rss_mb()
    print(f"  chunks:    {len(all_chunks)}")
    print(f"  time:      {_fmt_time(extract_time)}")
    print(f"  peak RSS:  {rss_after_extract:.0f} MiB (+{rss_after_extract - rss_start:.0f})")
    if files:
        print(f"  per file:  {_fmt_time(extract_time / len(files))}")
    if all_chunks:
        print(f"  per chunk: {_fmt_time(extract_time / len(all_chunks))}")
    print("  chunk kinds:")
    for kind, count in kind_dist.items():
        print(f"    {kind:20s} {count:>6d}")

    # ── Phase 3: DB insert ───────────────────────────────────────
    _section("DB inserts (chunks + snapshots)")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bench.duckdb"
        store = IndexStore(db_path)

        snapshot_rows = [(base_ref, e.path, e.blob_sha) for e in files]
        insert_time = _bench_db_insert(store, all_chunks, snapshot_rows)
        rss_after_insert = _rss_mb()
        db_size = db_path.stat().st_size
        print(f"  time:      {_fmt_time(insert_time)}")
        print(f"  db size:   {_fmt_size(db_size)}")
        print(f"  peak RSS:  {rss_after_insert:.0f} MiB (+{rss_after_insert - rss_start:.0f})")

        # ── Phase 4: Edge inference ──────────────────────────────
        _section("Edge inference")
        # Re-read chunks from store for edge inference (mirrors orchestrator).
        stored_chunks = store.get_chunks(base_ref)
        n_edges, edge_time = _bench_edge_inference(store, stored_chunks, files, base_ref)
        rss_after_edges = _rss_mb()
        print(f"  edges:     {n_edges}")
        print(f"  time:      {_fmt_time(edge_time)}")
        print(f"  peak RSS:  {rss_after_edges:.0f} MiB (+{rss_after_edges - rss_start:.0f})")

        # ── Phase 5: FTS index ───────────────────────────────────
        _section("FTS index rebuild")
        fts_time = _bench_fts(store)
        print(f"  time:      {_fmt_time(fts_time)}")

        # Checkpoint to flush.
        store.checkpoint()
        db_size_final = db_path.stat().st_size

        # ── Summary ──────────────────────────────────────────────
        total_time = list_time + extract_time + insert_time + edge_time + fts_time
        _section("Summary (no embedding)")
        print(f"  total time:    {_fmt_time(total_time)}")
        print(
            f"    listing:     {_fmt_time(list_time):>10s}  ({list_time / total_time * 100:4.1f}%)"
        )
        print(
            f"    extraction:  {_fmt_time(extract_time):>10s}  ({extract_time / total_time * 100:4.1f}%)"
        )
        print(
            f"    db insert:   {_fmt_time(insert_time):>10s}  ({insert_time / total_time * 100:4.1f}%)"
        )
        print(
            f"    edges:       {_fmt_time(edge_time):>10s}  ({edge_time / total_time * 100:4.1f}%)"
        )
        print(f"    fts rebuild: {_fmt_time(fts_time):>10s}  ({fts_time / total_time * 100:4.1f}%)")
        print(f"  files:         {len(files)}")
        print(f"  chunks:        {len(all_chunks)}")
        print(f"  edges:         {n_edges}")
        print(f"  db size:       {_fmt_size(db_size_final)}")
        print(f"  peak RSS:      {_rss_mb():.0f} MiB (delta +{_rss_mb() - rss_start:.0f})")

        # ── Query latency ────────────────────────────────────────
        _bench_queries(store, base_ref, len(all_chunks))

        # ── Full build_index (end-to-end, no embedding) ─────────
        _section("Full build_index (end-to-end, no embedding)")
        store.close()

        store2 = IndexStore(Path(tmp) / "bench2.duckdb")
        gc.collect()
        rss_before_build = _rss_mb()

        with patch("rbtr.index.orchestrator._embed_missing"):
            t0 = time.monotonic()
            result = build_index(repo, base_ref, store2)
            build_time = time.monotonic() - t0

        rss_after_build = _rss_mb()
        db2_size = (Path(tmp) / "bench2.duckdb").stat().st_size
        print(f"  time:      {_fmt_time(build_time)}")
        print(f"  chunks:    {result.stats.total_chunks}")
        print(f"  edges:     {result.stats.total_edges}")
        print(f"  parsed:    {result.stats.parsed_files}")
        print(f"  skipped:   {result.stats.skipped_files}")
        print(f"  errors:    {len(result.errors)}")
        print(f"  db size:   {_fmt_size(db2_size)}")
        print(f"  peak RSS:  {rss_after_build:.0f} MiB (+{rss_after_build - rss_before_build:.0f})")

        # ── Incremental update ───────────────────────────────────
        if head_ref:
            _section(f"Incremental update ({base_ref} → {head_ref})")

            from rbtr_legacy.index.git import changed_files

            changed = changed_files(repo, base_ref, head_ref)
            print(f"  changed files: {len(changed)}")

            with patch("rbtr.index.orchestrator._embed_missing"):
                t0 = time.monotonic()
                inc_result = update_index(repo, base_ref, head_ref, store2)
                inc_time = time.monotonic() - t0

            print(f"  time:      {_fmt_time(inc_time)}")
            print(f"  chunks:    {inc_result.stats.total_chunks}")
            print(f"  edges:     {inc_result.stats.total_edges}")
            print(f"  parsed:    {inc_result.stats.parsed_files}")
            print(f"  skipped:   {inc_result.stats.skipped_files}")

            _bench_queries(store2, head_ref, inc_result.stats.total_chunks)

        store2.close()

    print(f"\n{'─' * 64}")
    print(f"Final peak RSS: {_rss_mb():.0f} MiB")
    print("done.")


if __name__ == "__main__":
    main()
