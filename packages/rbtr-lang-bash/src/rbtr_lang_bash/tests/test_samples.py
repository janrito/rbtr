"""Bash sample extraction: the `samples/bash/` project, built as rbtr builds a repository and read as an agent reads it.

The snapshots are the golden record of what bash extraction produces;
regenerate with `pytest --snapshot-update` after an intended change.
Engine-wide invariants (determinism, line numbers, syntax-error recovery)
are covered once in core, not re-run per language.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_every_line_in_a_chunk, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "bash") as index:
        symbols = outline(index)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("bin/b.sh", "helper"): {("bin/a.sh", "source ./b.sh", "imports")},
        ("lib/colours.sh", "RED"): {("bash.sh", "source ./lib/colours.sh", "imports")},
        ("lib/colours.sh", "RESET"): {("bash.sh", "source ./lib/colours.sh", "imports")},
    }
    with indexed(Path(__file__).parent / "samples" / "bash") as index:
        assert_reached(index, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "bash") as index:
        assert_every_line_in_a_chunk(index)
