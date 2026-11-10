"""Ranking assertion helpers for search tests.

Each helper produces a clear failure message naming the chunks
involved so test output is immediately actionable. Chunks are
identified by `name`: an id is a content hash, so a literal one in a
test would say nothing about what failed.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl
from polars.testing import assert_frame_equal

from rbtr.domain.models import ChangeKind, ScoredChunk
from rbtr.index.results import ChangedSymbolRow

# A symbol's identity for diff assertions: (name, scope).
SymbolId = tuple[str, str]


def _rank_in(results: list[ScoredChunk], name: str) -> int | None:
    """Return 1-indexed rank of the chunk called *name*, or None."""
    for i, r in enumerate(results, 1):
        if r.name == name:
            return i
    return None


def assert_outranks(results: list[ScoredChunk], higher: str, lower: str) -> None:
    """Assert *higher* appears before *lower* in the ranking."""
    r_hi = _rank_in(results, higher)
    r_lo = _rank_in(results, lower)
    assert r_hi is not None, f"{higher} not found in results"
    assert r_lo is not None, f"{lower} not found in results"
    assert r_hi < r_lo, f"{higher} (rank {r_hi}) did not outrank {lower} (rank {r_lo})"


def assert_ranked_within(results: list[ScoredChunk], name: str, *, top: int) -> None:
    """Assert the chunk called *name* appears within the top *top*."""
    r = _rank_in(results, name)
    assert r is not None, f"{name} not found in results"
    assert r <= top, f"{name} ranked {r}, expected within top {top}"


def assert_in_results(results: list[ScoredChunk], name: str) -> ScoredChunk:
    """Assert a chunk called *name* is present and return it."""
    for r in results:
        if r.name == name:
            return r
    msg = f"{name} not found in results"
    raise AssertionError(msg)


def assert_changes(
    frame: dy.DataFrame[ChangedSymbolRow],
    *,
    added: set[SymbolId],
    modified: set[SymbolId],
    removed: set[SymbolId],
) -> None:
    """Assert a `changed_symbols` result holds exactly these symbols.

    Compares the full added/modified/removed sets, so an unexpected
    extra symbol (e.g. an unchanged neighbour leaking in) fails just
    as loudly as a missing one.  A symbol listed twice is
    `assert_no_duplicate_changes`'s to catch.
    """
    expected = pl.DataFrame(
        [
            {"name": name, "scope": scope, "change_kind": change.value}
            for change, symbols in (
                (ChangeKind.ADDED, added),
                (ChangeKind.MODIFIED, modified),
                (ChangeKind.REMOVED, removed),
            )
            for name, scope in symbols
        ],
        schema={"name": pl.String, "scope": pl.String, "change_kind": pl.String},
    )
    actual = frame.select("name", "scope", pl.col("change_kind").cast(pl.String)).unique()
    assert_frame_equal(actual, expected, check_row_order=False)


def assert_no_duplicate_changes(frame: dy.DataFrame[ChangedSymbolRow]) -> None:
    """Assert each changed symbol appears at most once per bucket.

    A legitimate diff emits one row per head (or base) symbol. A
    duplicate `(file_path, name, scope, change_kind)` row means the
    `modified` branch has fanned out across a non-unique identity (the
    NxM cross-join bug). Counting stays in polars rather than a
    hand-rolled Python tally.
    """
    duplicates = (
        frame.group_by("file_path", "name", "scope", "change_kind").len().filter(pl.col("len") > 1)
    )
    assert duplicates.is_empty(), f"duplicate changed symbols (identity fan-out):\n{duplicates}"
