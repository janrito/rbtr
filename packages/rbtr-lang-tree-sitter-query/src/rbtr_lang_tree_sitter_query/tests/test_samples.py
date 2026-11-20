"""tree-sitter query sample extraction: `samples/tree_sitter_query/` built as rbtr builds a repository and read as an agent reads it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_every_line_in_a_chunk, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "tree_sitter_query") as index:
        symbols = outline(index)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached: dict[Target, set[Referrer]] = {}
    with indexed(Path(__file__).parent / "samples" / "tree_sitter_query") as index:
        assert_reached(index, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "tree_sitter_query") as index:
        assert_every_line_in_a_chunk(index)
