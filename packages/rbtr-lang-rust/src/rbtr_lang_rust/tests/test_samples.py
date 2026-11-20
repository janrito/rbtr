"""rust sample extraction: the `samples/rust/` project, built as rbtr builds a repository and read as an agent reads it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_every_line_in_a_chunk, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "rust") as index:
        symbols = outline(index)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("rust.rs", "greet"): {("rust.rs", "greet_world", "calls")},
        ("src/config.rs", "LOCALE"): {
            ("rust.rs", "use crate::config::LOCALE;", "imports"),
            ("src/orders.rs", "use crate::config::LOCALE as locale;", "imports"),
        },
        ("src/orders.rs", "new"): {("src/orders.rs", "build", "calls")},
        ("src/orders.rs", "valid"): {("src/orders.rs", "check", "calls")},
    }
    with indexed(Path(__file__).parent / "samples" / "rust") as index:
        assert_reached(index, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "rust") as index:
        assert_every_line_in_a_chunk(index)
