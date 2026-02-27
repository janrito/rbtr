"""Round-trip tests for ``_position_to_line`` / ``_line_to_position``.

Data-first: each ``DIFF_HUNK`` is a realistic diff excerpt (taken from
real Git output or modelled on GitHub API responses).  Every
``(position, line, side)`` triple in the hunk is enumerated so we can
verify both conversion directions and prove they are inverses.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rbtr.github.client import _line_to_position, _position_to_line

# ── Shared diff data ─────────────────────────────────────────────────


@dataclass(frozen=True)
class HunkPoint:
    """A single commentable point within a diff hunk."""

    position: int
    line: int
    side: str


@dataclass(frozen=True)
class DiffFixture:
    """A diff hunk and every commentable point inside it."""

    id: str
    diff_hunk: str
    points: tuple[HunkPoint, ...]


# ── Dataset ──────────────────────────────────────────────────────────
#
# Each fixture includes the FULL hunk (header through last line) and
# exhaustively enumerates (position, line, side) for every diff line.

DIFF_FIXTURES: list[DiffFixture] = [
    # 1. Brand-new file — every line is an addition.
    DiffFixture(
        id="new-file",
        diff_hunk=("@@ -0,0 +1,5 @@\n+import os\n+import sys\n+\n+def main():\n+    pass"),
        points=(
            HunkPoint(position=1, line=1, side="RIGHT"),
            HunkPoint(position=2, line=2, side="RIGHT"),
            HunkPoint(position=3, line=3, side="RIGHT"),
            HunkPoint(position=4, line=4, side="RIGHT"),
            HunkPoint(position=5, line=5, side="RIGHT"),
        ),
    ),
    # 2. Deleted file — every line is a deletion.
    DiffFixture(
        id="deleted-file",
        diff_hunk=("@@ -1,3 +0,0 @@\n-# old module\n-import legacy\n-legacy.run()"),
        points=(
            HunkPoint(position=1, line=1, side="LEFT"),
            HunkPoint(position=2, line=2, side="LEFT"),
            HunkPoint(position=3, line=3, side="LEFT"),
        ),
    ),
    # 3. Simple replacement — context + deletion + addition + context.
    #    Modelled on a real one-line fix.
    DiffFixture(
        id="simple-replacement",
        diff_hunk=(
            "@@ -10,5 +10,5 @@\n"
            " def process(data):\n"
            "     result = []\n"
            "-    for item in data:\n"
            "+    for item in sorted(data):\n"
            "         result.append(item)"
        ),
        points=(
            #                          old  new
            # " def process(data):"    10   10   context
            # "     result = []"       11   11   context
            # "-    for item …"        12        deletion
            # "+    for item …"             12   addition
            # "         result…"       13   13   context
            HunkPoint(position=1, line=10, side="RIGHT"),  # context
            HunkPoint(position=2, line=11, side="RIGHT"),  # context
            HunkPoint(position=3, line=12, side="LEFT"),  # deletion
            HunkPoint(position=4, line=12, side="RIGHT"),  # addition
            HunkPoint(position=5, line=13, side="RIGHT"),  # context
        ),
    ),
    # 4. Multi-line deletion — lines removed without replacement.
    DiffFixture(
        id="multi-line-deletion",
        diff_hunk=(
            "@@ -20,6 +20,3 @@\n"
            "     # keep this\n"
            "-    debug_log('a')\n"
            "-    debug_log('b')\n"
            "-    debug_log('c')\n"
            "     return True"
        ),
        points=(
            #                       old  new
            # " # keep this"        20   20   context
            # "- debug_log('a')"    21        deletion
            # "- debug_log('b')"    22        deletion
            # "- debug_log('c')"    23        deletion
            # " return True"        24   21   context
            HunkPoint(position=1, line=20, side="RIGHT"),  # context
            HunkPoint(position=2, line=21, side="LEFT"),  # deletion
            HunkPoint(position=3, line=22, side="LEFT"),  # deletion
            HunkPoint(position=4, line=23, side="LEFT"),  # deletion
            HunkPoint(position=5, line=21, side="RIGHT"),  # context
        ),
    ),
    # 5. Multi-line addition — new block inserted.
    DiffFixture(
        id="multi-line-addition",
        diff_hunk=(
            "@@ -5,3 +5,7 @@\n"
            " class Config:\n"
            "+    timeout: int = 30\n"
            "+    retries: int = 3\n"
            "+    backoff: float = 1.5\n"
            "+\n"
            "     name: str = ''"
        ),
        points=(
            #                       old  new
            # " class Config:"      5    5    context
            # "+ timeout…"               6    addition
            # "+ retries…"               7    addition
            # "+ backoff…"               8    addition
            # "+"                         9    addition (blank line)
            # "     name…"          6    10   context
            HunkPoint(position=1, line=5, side="RIGHT"),  # context
            HunkPoint(position=2, line=6, side="RIGHT"),  # addition
            HunkPoint(position=3, line=7, side="RIGHT"),  # addition
            HunkPoint(position=4, line=8, side="RIGHT"),  # addition
            HunkPoint(position=5, line=9, side="RIGHT"),  # addition
            HunkPoint(position=6, line=10, side="RIGHT"),  # context
        ),
    ),
    # 6. Interleaved changes — deletions and additions mixed with context.
    #    Real-world pattern: refactoring a function signature.
    DiffFixture(
        id="interleaved-changes",
        diff_hunk=(
            "@@ -40,9 +40,10 @@\n"
            " def fetch(\n"
            "-    url: str,\n"
            "-    timeout: int = 30,\n"
            "+    url: str | URL,\n"
            "+    timeout: float = 30.0,\n"
            "+    retries: int = 3,\n"
            " ) -> Response:\n"
            "-    return _get(url, timeout)\n"
            "+    return _get(url, timeout, retries)"
        ),
        points=(
            #                           old  new
            # " def fetch("             40   40   context
            # "- url: str,"             41        deletion
            # "- timeout: int…"         42        deletion
            # "+ url: str | URL,"            41   addition
            # "+ timeout: float…"            42   addition
            # "+ retries: int…"              43   addition
            # " ) -> Response:"         43   44   context
            # "- return _get(…)"        44        deletion
            # "+ return _get(…)"             45   addition
            HunkPoint(position=1, line=40, side="RIGHT"),  # context
            HunkPoint(position=2, line=41, side="LEFT"),  # deletion
            HunkPoint(position=3, line=42, side="LEFT"),  # deletion
            HunkPoint(position=4, line=41, side="RIGHT"),  # addition
            HunkPoint(position=5, line=42, side="RIGHT"),  # addition
            HunkPoint(position=6, line=43, side="RIGHT"),  # addition
            HunkPoint(position=7, line=44, side="RIGHT"),  # context
            HunkPoint(position=8, line=44, side="LEFT"),  # deletion
            HunkPoint(position=9, line=45, side="RIGHT"),  # addition
        ),
    ),
    # 7. No-newline marker at end of file.
    DiffFixture(
        id="no-newline-eof",
        diff_hunk=(
            "@@ -1,2 +1,2 @@\n"
            " first\n"
            "-second\n"
            "\\ No newline at end of file\n"
            "+second_v2\n"
            "\\ No newline at end of file"
        ),
        points=(
            HunkPoint(position=1, line=1, side="RIGHT"),  # context
            HunkPoint(position=2, line=2, side="LEFT"),  # deletion
            # position 3 is "\ No newline" — skipped
            HunkPoint(position=3, line=2, side="RIGHT"),  # addition
            # position 4 is "\ No newline" — skipped
        ),
    ),
    # 8. Hunk starting at line 1 with no old count (short form).
    DiffFixture(
        id="short-header",
        diff_hunk="@@ -1 +1,2 @@\n first_line\n+inserted",
        points=(
            HunkPoint(position=1, line=1, side="RIGHT"),  # context
            HunkPoint(position=2, line=2, side="RIGHT"),  # addition
        ),
    ),
    # 9. Large offset — hunk deep in a big file.
    DiffFixture(
        id="large-offset",
        diff_hunk=(
            "@@ -500,4 +510,5 @@\n"
            "     return cache[key]\n"
            "+    logger.debug('cache hit')\n"
            "     hits += 1\n"
            "     total += 1"
        ),
        points=(
            #                          old   new
            # " return cache[key]"     500   510  context
            # "+ logger.debug(…)"            511  addition
            # " hits += 1"             501   512  context
            # " total += 1"            502   513  context
            HunkPoint(position=1, line=510, side="RIGHT"),
            HunkPoint(position=2, line=511, side="RIGHT"),
            HunkPoint(position=3, line=512, side="RIGHT"),
            HunkPoint(position=4, line=513, side="RIGHT"),
        ),
    ),
    # 10. Real hunk from this repo (client.py logging addition).
    DiffFixture(
        id="real-rbtr-logging",
        diff_hunk=(
            "@@ -2,6 +2,7 @@\n"
            " \n"
            " from __future__ import annotations\n"
            " \n"
            "+import logging\n"
            " import re\n"
            " from collections import Counter"
        ),
        points=(
            HunkPoint(position=1, line=2, side="RIGHT"),
            HunkPoint(position=2, line=3, side="RIGHT"),
            HunkPoint(position=3, line=4, side="RIGHT"),
            HunkPoint(position=4, line=5, side="RIGHT"),  # +import logging
            HunkPoint(position=5, line=6, side="RIGHT"),
            HunkPoint(position=6, line=7, side="RIGHT"),
        ),
    ),
]


# ── Helpers ──────────────────────────────────────────────────────────


def _fixture_ids() -> list[str]:
    return [f.id for f in DIFF_FIXTURES]


def _all_points() -> list[pytest.param]:
    """Expand fixtures into parametrize entries with descriptive ids."""
    out: list[pytest.param] = []
    for f in DIFF_FIXTURES:
        for p in f.points:
            out.append(
                pytest.param(
                    f.id,
                    f.diff_hunk,
                    p,
                    id=f"{f.id}-pos{p.position}-{p.side}{p.line}",
                )
            )
    return out


# ── position → (line, side) ──────────────────────────────────────────


@pytest.mark.parametrize(("fixture_id", "diff_hunk", "point"), _all_points())
def test_position_to_line(fixture_id: str, diff_hunk: str, point: HunkPoint) -> None:
    """The diff_hunk truncated to *position* lines recovers (line, side)."""
    # Build the truncated hunk that GitHub would return for this position.
    lines = diff_hunk.split("\n")
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("@@"))
    # Count real diff lines (skip "\ No newline" markers).
    kept: list[str] = lines[: header_idx + 1]
    real_count = 0
    for ln in lines[header_idx + 1 :]:
        kept.append(ln)
        if not ln.startswith("\\"):
            real_count += 1
        if real_count == point.position:
            break
    truncated = "\n".join(kept)

    result_line, result_side = _position_to_line(truncated)
    assert result_line == point.line, f"line: expected {point.line}, got {result_line}"
    assert result_side == point.side, f"side: expected {point.side}, got {result_side}"


# ── (line, side) → position ──────────────────────────────────────────


@pytest.mark.parametrize(("fixture_id", "diff_hunk", "point"), _all_points())
def test_line_to_position(fixture_id: str, diff_hunk: str, point: HunkPoint) -> None:
    """(line, side) maps back to the expected diff position."""
    result_pos = _line_to_position(diff_hunk, point.line, point.side)
    assert result_pos == point.position, f"position: expected {point.position}, got {result_pos}"


# ── Round-trip: position → line → position ────────────────────────────


@pytest.mark.parametrize(("fixture_id", "diff_hunk", "point"), _all_points())
def test_roundtrip_position_line_position(
    fixture_id: str, diff_hunk: str, point: HunkPoint
) -> None:
    """position → (line, side) → position is an identity."""
    # Truncate to position (as GitHub would).
    lines = diff_hunk.split("\n")
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("@@"))
    kept: list[str] = lines[: header_idx + 1]
    real_count = 0
    for ln in lines[header_idx + 1 :]:
        kept.append(ln)
        if not ln.startswith("\\"):
            real_count += 1
        if real_count == point.position:
            break
    truncated = "\n".join(kept)

    line, side = _position_to_line(truncated)
    pos = _line_to_position(diff_hunk, line, side)
    assert pos == point.position, (
        f"roundtrip failed: pos {point.position} → ({line}, {side}) → {pos}"
    )


# ── Edge cases ────────────────────────────────────────────────────────


def test_empty_hunk_returns_zero() -> None:
    assert _position_to_line("") == (0, "RIGHT")


def test_malformed_header_returns_zero() -> None:
    assert _position_to_line("not a hunk\n+added") == (0, "RIGHT")


def test_line_not_in_hunk_returns_none() -> None:
    hunk = "@@ -1,2 +1,2 @@\n first\n-second"
    assert _line_to_position(hunk, 999, "RIGHT") is None


def test_wrong_side_returns_none() -> None:
    hunk = "@@ -1,2 +1,2 @@\n first\n+added"
    # Line 2 is on RIGHT (addition), not LEFT.
    assert _line_to_position(hunk, 2, "LEFT") is None
