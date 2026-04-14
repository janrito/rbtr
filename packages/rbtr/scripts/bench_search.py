#!/usr/bin/env python3
"""Mine session history for real search queries and replay them.

Extracts search→read_symbol pairs from the sessions database,
replays each query through the current unified `search()`
pipeline, and reports how often the actually-read symbol appears
in the top results.

Usage::

    just bench-search                     # current directory
    just bench-search -- /path/to/repo    # specific repo

Only events matching the given repo (by remote URL) are
replayed; events from other repos are skipped.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pygit2

from rbtr.index import orchestrator
from rbtr.index.orchestrator import build_index
from rbtr.index.search import ScoredResult
from rbtr.index.store import IndexStore

# ── Data types ───────────────────────────────────────────────────────

_SEARCH_TOOLS = frozenset({"search_symbols", "search_codebase", "search_similar", "search"})


@dataclass
class SearchEvent:
    """A single historical search call with optional relevance label."""

    query: str
    original_tool: str
    session_id: str
    repo_owner: str | None = None
    repo_name: str | None = None
    read_target: str | None = None  # name arg of subsequent read_symbol
    is_retry: bool = False
    first_query_in_chain: str | None = None


@dataclass
class ReplayResult:
    """Result of replaying a SearchEvent against the current search."""

    event: SearchEvent
    rank: int | None = None  # rank of read_target, None if not found
    results: list[ScoredResult] = field(default_factory=list)
    target_result: ScoredResult | None = None  # ScoredResult for the target


# ── Extraction ───────────────────────────────────────────────────────


def _parse_args(data_json: str) -> dict[str, Any]:
    """Extract the tool call arguments from a fragment's data_json.

    The `args` field is itself a JSON-encoded string inside the
    outer JSON object.  Returns untyped dict (JSON boundary).
    """
    outer = json.loads(data_json)
    raw_args = outer.get("args", "{}")
    if isinstance(raw_args, str):
        return json.loads(raw_args)
    return raw_args


def _extract_query(tool_name: str, args: dict[str, Any]) -> str | None:
    """Normalise the query string across old and new tool signatures."""
    if tool_name == "search_symbols":
        return args.get("name")
    # search_codebase, search_similar, search — all use "query"
    return args.get("query")


def _parse_search_result_names(content: str) -> set[str]:
    """Extract symbol names from a search tool-return content string.

    Search results have lines like::

        function add_user_prompt  (path/to/file.py:42)
        [19.54] function test_foo  (path/to/file.py:10)

    We extract the second word (the symbol name) from each line.
    """
    names: set[str] = set()
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip BM25 score prefix like "[19.54] ".
        if line.startswith("["):
            bracket_end = line.find("] ")
            if bracket_end != -1:
                line = line[bracket_end + 2 :]
        parts = line.split()
        if len(parts) >= 2:
            names.add(parts[1])
    return names


def extract_events(db_path: Path) -> list[SearchEvent]:
    """Scan the sessions DB for search calls and pair with read_symbol.

    For each search tool-call, looks for a `read_symbol` tool-call
    in the same session that occurs *after* the search but *before*
    the next user-prompt or unrelated tool chain.  Only pairs where
    the read target is *directly* after the search (no more than 2
    intervening tool-calls) or was present in the search results.
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    # Fetch tool-calls, tool-returns, and user-prompts.
    rows = con.execute(
        """
        SELECT id, session_id, tool_name, data_json, created_at,
               fragment_kind, repo_owner, repo_name
        FROM fragments
        WHERE fragment_kind IN ('tool-call', 'tool-return', 'user-prompt')
          AND (tool_name IS NOT NULL OR fragment_kind = 'user-prompt')
        ORDER BY session_id, created_at, fragment_index
        """
    ).fetchall()
    con.close()

    events: list[SearchEvent] = []

    # Group rows by session for linear scanning.
    sessions: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        sid = row["session_id"]
        sessions.setdefault(sid, []).append(row)

    for sid, fragments in sessions.items():
        i = 0
        while i < len(fragments):
            frag = fragments[i]

            # Only process search tool-calls.
            if frag["fragment_kind"] != "tool-call" or frag["tool_name"] not in _SEARCH_TOOLS:
                i += 1
                continue

            args = _parse_args(frag["data_json"])
            query = _extract_query(frag["tool_name"], args)
            if not query:
                i += 1
                continue

            # Collect search result names from the tool-return.
            result_names: set[str] = set()
            # Count non-return tool-calls between search and read.
            intervening_calls = 0
            read_target: str | None = None

            j = i + 1
            while j < len(fragments):
                nxt = fragments[j]
                if nxt["fragment_kind"] == "user-prompt":
                    break

                # Collect result names from the search return.
                if nxt["fragment_kind"] == "tool-return" and nxt["tool_name"] in _SEARCH_TOOLS:
                    data = json.loads(nxt["data_json"])
                    content = data.get("content", "")
                    if isinstance(content, str):
                        result_names = _parse_search_result_names(content)

                # Found a read_symbol — check if it's related.
                if nxt["fragment_kind"] == "tool-call" and nxt["tool_name"] == "read_symbol":
                    nxt_args = _parse_args(nxt["data_json"])
                    candidate = nxt_args.get("name")
                    # Pair if: close to the search OR was in the search results.
                    if candidate and (intervening_calls <= 2 or candidate in result_names):
                        read_target = candidate
                    break

                # Count intervening tool-calls (not returns).
                if nxt["fragment_kind"] == "tool-call":
                    intervening_calls += 1
                    # After many unrelated calls without result names,
                    # stop looking — the read is likely unrelated.
                    if intervening_calls > 10:
                        break

                j += 1

            events.append(
                SearchEvent(
                    query=query,
                    original_tool=frag["tool_name"],
                    session_id=sid,
                    repo_owner=frag["repo_owner"],
                    repo_name=frag["repo_name"],
                    read_target=read_target,
                )
            )
            i += 1

    # Detect retry chains: consecutive searches in the same session
    # with overlapping query terms.
    _detect_retries(events)

    return events


def _word_set(text: str) -> set[str]:
    """Split text into a lowercase word set for Jaccard comparison."""
    return {w.lower().strip("(") for w in text.split() if w.strip("(")}


def _detect_retries(events: list[SearchEvent]) -> None:
    """Mark consecutive searches with overlapping terms as retries."""
    prev: SearchEvent | None = None
    chain_start: SearchEvent | None = None
    for ev in events:
        if prev and prev.session_id == ev.session_id:
            a, b = _word_set(prev.query), _word_set(ev.query)
            union = a | b
            if union and len(a & b) / len(union) > 0.3:
                if chain_start is None:
                    chain_start = prev
                ev.is_retry = True
                ev.first_query_in_chain = chain_start.query
            else:
                chain_start = None
        else:
            chain_start = None
        prev = ev


# ── Replay ───────────────────────────────────────────────────────────


def _find_target_rank(
    results: list[ScoredResult], target_name: str
) -> tuple[int | None, ScoredResult | None]:
    """Find the rank (1-indexed) of *target_name* in results."""
    for i, r in enumerate(results, 1):
        if r.chunk.name == target_name:
            return i, r
    return None, None


def replay_events(
    events: list[SearchEvent],
    store: IndexStore,
    commit_sha: str,
    top_k: int = 10,
) -> list[ReplayResult]:
    """Replay search events against the current search pipeline."""
    replays: list[ReplayResult] = []
    for ev in events:
        results = store.search(commit_sha, ev.query, top_k=top_k)
        rank, target_sr = (None, None)
        if ev.read_target:
            rank, target_sr = _find_target_rank(results, ev.read_target)
        replays.append(
            ReplayResult(
                event=ev,
                rank=rank,
                results=results,
                target_result=target_sr,
            )
        )
    return replays


# ── Report ───────────────────────────────────────────────────────────


def _print_header(title: str) -> None:
    print(f"\n{'═' * 4} {title} {'═' * (60 - len(title))}")


def _print_aggregate(replays: list[ReplayResult]) -> None:
    """Print aggregate metrics for search→read pairs."""
    with_target = [r for r in replays if r.event.read_target]
    without_target = [r for r in replays if not r.event.read_target]

    total = len(replays)
    n_paired = len(with_target)
    n_unpaired = len(without_target)

    n_retries = sum(1 for r in replays if r.event.is_retry)

    print(f"  Total search calls:     {total}")
    print(f"  With read_symbol:       {n_paired}")
    print(f"  Without (abandoned):    {n_unpaired}")
    print(f"  Retry chains:           {n_retries}")
    print()

    if not with_target:
        print("  No search→read pairs to evaluate.")
        return

    r1 = sum(1 for r in with_target if r.rank == 1)
    r5 = sum(1 for r in with_target if r.rank is not None and r.rank <= 5)
    mrr = sum(1.0 / r.rank for r in with_target if r.rank is not None)

    print(f"  Search→read pairs:      {n_paired}")
    print(f"    Result at rank 1:     {r1:>3} ({r1 / n_paired:.0%})")
    print(f"    Result in top 5:      {r5:>3} ({r5 / n_paired:.0%})")
    print(f"    MRR:                  {mrr / n_paired:.3f}")
    print()

    # Show misses
    misses = [r for r in with_target if r.rank is None or r.rank > 5]
    if misses:
        print("  Misses (target not in top 5):")
        for r in misses:
            rank_str = str(r.rank) if r.rank else "not found"
            print(f"    q={r.event.query!r:<45} target={r.event.read_target!r}")
            print(f"      tool={r.event.original_tool}  rank={rank_str}")
            if r.results:
                top3 = [
                    f"{sr.chunk.file_path.split('/')[-1]}:{sr.chunk.name}" for sr in r.results[:3]
                ]
                print(f"      got: {', '.join(top3)}")
            print()


def _print_detail(replays: list[ReplayResult]) -> None:
    """Print per-query signal breakdown for misranked results."""
    with_target = [r for r in replays if r.event.read_target]
    misranked = [r for r in with_target if r.rank != 1]

    if not misranked:
        print("  All targets ranked #1 — no signal breakdown needed.")
        return

    for r in misranked:
        rank_str = str(r.rank) if r.rank else "not found"
        print(f"  query: {r.event.query!r}")
        print(f"  target: {r.event.read_target} (rank: {rank_str})")

        if r.target_result:
            sr = r.target_result
            print(
                f"    lex={sr.lexical:.2f}  sem={sr.semantic:.2f}"
                f"  name={sr.name:.2f}  kind={sr.kind_boost:.1f}"
                f"  file={sr.file_penalty:.1f}  imp={sr.importance:.2f}"
                f"  prox={sr.proximity:.1f}  FINAL={sr.score:.3f}"
            )

        # Compare with rank #1 if different.
        if r.results and (r.rank is None or r.rank > 1):
            top = r.results[0]
            print(f"  rank #1: {top.chunk.file_path}:{top.chunk.name}")
            print(
                f"    lex={top.lexical:.2f}  sem={top.semantic:.2f}"
                f"  name={top.name:.2f}  kind={top.kind_boost:.1f}"
                f"  file={top.file_penalty:.1f}  imp={top.importance:.2f}"
                f"  prox={top.proximity:.1f}  FINAL={top.score:.3f}"
            )

        print()


def _print_query_distribution(events: list[SearchEvent]) -> None:
    """Show what kinds of queries the LLM actually generates."""
    from rbtr.index.search import QueryKind, classify_query

    counts: dict[QueryKind, int] = {}
    for ev in events:
        kind = classify_query(ev.query)
        counts[kind] = counts.get(kind, 0) + 1

    print("  Query classification:")
    for kind in QueryKind:
        n = counts.get(kind, 0)
        pct = n / len(events) * 100 if events else 0
        print(f"    {kind.value:<12} {n:>3} ({pct:4.0f}%)")
    print()

    # Show all queries grouped by classification
    print("  All queries:")
    for ev in events:
        kind = classify_query(ev.query)
        target = f" → {ev.read_target}" if ev.read_target else ""
        retry = " (retry)" if ev.is_retry else ""
        print(f"    [{kind.value[0]}] {ev.query!r}{target}{retry}")
    print()


# ── Repo matching ────────────────────────────────────────────────────


def _repo_matches(repo: pygit2.Repository, repo_key: str) -> bool:
    """Check if *repo* matches *repo_key* (`owner/name`)."""
    repo_name = repo_key.split("/")[-1] if "/" in repo_key else repo_key
    remote_str = " ".join(rm.url for rm in repo.remotes).lower()
    return repo_name.lower() in remote_str


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    sessions_db = Path("~/.config/rbtr/sessions.db").expanduser()
    args = [a for a in sys.argv[1:] if a != "--"]
    repo_path = Path(args[0]) if args else Path.cwd()

    if not sessions_db.exists():
        print(f"Sessions DB not found: {sessions_db}", file=sys.stderr)
        sys.exit(1)

    try:
        repo = pygit2.Repository(str(repo_path))
    except Exception:
        print(f"Not a git repo: {repo_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Sessions DB: {sessions_db}")
    print(f"Repo:        {repo_path}")

    events = extract_events(sessions_db)
    print(f"Extracted {len(events)} search events")

    if not events:
        print("No search events found.")
        return

    # Filter to events matching the current repo.
    repo_events: list[SearchEvent] = []
    skipped_repos: set[str] = set()
    for ev in events:
        key = f"{ev.repo_owner}/{ev.repo_name}" if ev.repo_owner else "unknown"
        if _repo_matches(repo, key):
            repo_events.append(ev)
        else:
            skipped_repos.add(key)

    if skipped_repos:
        print(f"Skipped {len(skipped_repos)} other repo(s): {', '.join(sorted(skipped_repos))}")

    if not repo_events:
        print("No search events for this repo.")
        return

    _print_header(f"Searches ({len(repo_events)})")
    _print_query_distribution(repo_events)

    paired = [ev for ev in repo_events if ev.read_target]
    if not paired:
        print("  No search→read pairs — nothing to replay.\n")
        return

    sha = str(repo.revparse_single("HEAD").id)
    print(f"  Building index for HEAD ({sha[:8]})...")
    with tempfile.TemporaryDirectory() as tmp:
        store = IndexStore(Path(tmp) / "bench.duckdb")
        # Skip embeddings — we only need structural + lexical + name signals.
        _real_embed = orchestrator._embed_missing
        orchestrator._embed_missing = lambda *a, **kw: None  # type: ignore[assignment]
        try:
            build_index(repo, sha, store)
        finally:
            orchestrator._embed_missing = _real_embed  # type: ignore[assignment]

        replays = replay_events(repo_events, store, sha)

        _print_header("Aggregate")
        _print_aggregate(replays)

        _print_header("Signal breakdown (misranked)")
        _print_detail(replays)

        store.close()


if __name__ == "__main__":
    main()
