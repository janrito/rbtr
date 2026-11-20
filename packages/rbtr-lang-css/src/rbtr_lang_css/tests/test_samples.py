"""CSS sample extraction: the `samples/css/` project, built as rbtr builds a repository and read as an agent reads it.

The snapshots are the golden record of what CSS extraction produces.
Engine-wide invariants are covered once in core.
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
    with indexed(Path(__file__).parent / "samples" / "css") as index:
        symbols = outline(index)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("partials/dark.css", ".greeting--dark"): {
            ("css.css", '@import "partials/dark.css";', "imports")
        },
        ("partials/palette.css", "--ink"): {
            ("partials/dark.css", '@import "palette.css";', "imports")
        },
        ("partials/palette.css", ":root"): {
            ("partials/dark.css", '@import "palette.css";', "imports")
        },
        ("reset.css", "*"): {("css.css", '@import url("reset.css");', "imports")},
        ("theme.css", "--greeting-color"): {("css.css", '@import "theme.css";', "imports")},
        ("theme.css", ":root"): {("css.css", '@import "theme.css";', "imports")},
    }
    with indexed(Path(__file__).parent / "samples" / "css") as index:
        assert_reached(index, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "css") as index:
        assert_every_line_in_a_chunk(index)
