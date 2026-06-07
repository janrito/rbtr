"""Ranking assertion helpers for search tests.

Each helper produces a clear failure message naming the chunk
IDs involved so test output is immediately actionable.
"""

from __future__ import annotations

from rbtr.index.models import ChangeKind, Chunk, ScoredChunk

# A symbol's identity for diff assertions: (name, scope).
SymbolId = tuple[str, str]

# ── Private projections ─────────────────────────────────────────────


def _rank_in(results: list[ScoredChunk], chunk_id: str) -> int | None:
    """Return 1-indexed rank of *chunk_id*, or None if absent."""
    for i, r in enumerate(results, 1):
        if r.id == chunk_id:
            return i
    return None


def _ids_of(results: list[ScoredChunk]) -> list[str]:
    """Extract chunk IDs in rank order."""
    return [r.id for r in results]


# ── Public assertion helpers ────────────────────────────────────────


def assert_outranks(results: list[ScoredChunk], higher: str, lower: str) -> None:
    """Assert *higher* appears before *lower* in the ranking."""
    r_hi = _rank_in(results, higher)
    r_lo = _rank_in(results, lower)
    assert r_hi is not None, f"{higher} not found in results"
    assert r_lo is not None, f"{lower} not found in results"
    assert r_hi < r_lo, f"{higher} (rank {r_hi}) did not outrank {lower} (rank {r_lo})"


def assert_ranked_within(results: list[ScoredChunk], chunk_id: str, *, top: int) -> None:
    """Assert *chunk_id* appears within the top *top* results."""
    r = _rank_in(results, chunk_id)
    assert r is not None, f"{chunk_id} not found in results"
    assert r <= top, f"{chunk_id} ranked {r}, expected within top {top}"


def assert_in_results(results: list[ScoredChunk], chunk_id: str) -> ScoredChunk:
    """Assert *chunk_id* is present and return its `ScoredChunk`."""
    for r in results:
        if r.id == chunk_id:
            return r
    msg = f"{chunk_id} not found in results"
    raise AssertionError(msg)


def assert_same_ranking(a: list[ScoredChunk], b: list[ScoredChunk]) -> None:
    """Assert two result lists have identical chunk ID ordering."""
    ids_a = _ids_of(a)
    ids_b = _ids_of(b)
    assert ids_a == ids_b, f"rankings differ: {ids_a} != {ids_b}"


def assert_changes(
    result: list[tuple[Chunk, ChangeKind]],
    *,
    added: set[SymbolId],
    modified: set[SymbolId],
    removed: set[SymbolId],
) -> None:
    """Assert a `diff_symbols` result holds exactly these symbols.

    Compares the full added/modified/removed sets, so an unexpected
    extra symbol (e.g. an unchanged neighbour leaking in) fails just
    as loudly as a missing one.
    """
    buckets: dict[ChangeKind, set[SymbolId]] = {kind: set() for kind in ChangeKind}
    for chunk, change in result:
        buckets[change].add((chunk.name, chunk.scope))
    assert buckets[ChangeKind.ADDED] == added
    assert buckets[ChangeKind.MODIFIED] == modified
    assert buckets[ChangeKind.REMOVED] == removed
