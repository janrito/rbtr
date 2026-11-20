"""Less sample extraction: the `samples/less/` project, built as rbtr builds a repository and read as an agent reads it.

The snapshots are the golden record of what Less extraction produces.
Engine-wide invariants are covered once in core.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "less") as build:
        symbols = outline(build)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("base.less", "@bg"): {("less.less", '@import "base";', "imports")},
        ("partials/dark.less", ".greeting--dark"): {
            ("less.less", '@import "partials/dark.less";', "imports")
        },
        ("partials/tokens.less", "@ink"): {
            ("partials/dark.less", '@import "tokens.less";', "imports")
        },
    }
    with indexed(Path(__file__).parent / "samples" / "less") as build:
        assert_reached(build, reached)
