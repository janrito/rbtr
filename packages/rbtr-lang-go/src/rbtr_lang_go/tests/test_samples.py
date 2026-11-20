"""go sample extraction: the `samples/go/` project, built as rbtr builds a repository and read as an agent reads it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "go") as build:
        symbols = outline(build)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("go.go", "Greet"): {("go.go", "FormatGreeting", "calls")},
        ("greeter/text.go", "Trim"): {
            ("go.go", '"example.com/greeter/greeter"', "imports"),
            ("go.go", "Greet", "calls"),
        },
        ("shop/model.go", "Model"): {("shop/order.go", "Order", "inherits")},
        ("shop/model.go", "valid"): {
            ("shop/model.go", "Check", "calls"),
            ("shop/order.go", "NewOrder", "calls"),
        },
    }
    with indexed(Path(__file__).parent / "samples" / "go") as build:
        assert_reached(build, reached)
