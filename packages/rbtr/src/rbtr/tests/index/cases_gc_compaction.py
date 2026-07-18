"""Scenarios for the `gc` compaction flag.

Each scenario runs `handle_gc` with a given `compact` flag against a
churned, file-backed index and states whether the on-disk file should
shrink.  Compaction and `--no-compact` share the same assertion logic
(measure size before/after, then check data survives), differing only
in this flag — so they are cases of one test, not two tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompactionScenario:
    compact: bool
    expect_shrink: bool


def case_compact_reclaims() -> CompactionScenario:
    """Default compaction rewrites the file, returning freed blocks to the OS."""
    return CompactionScenario(compact=True, expect_shrink=True)


def case_no_compact_preserves() -> CompactionScenario:
    """`--no-compact` skips the rewrite, so the on-disk file is untouched."""
    return CompactionScenario(compact=False, expect_shrink=False)
