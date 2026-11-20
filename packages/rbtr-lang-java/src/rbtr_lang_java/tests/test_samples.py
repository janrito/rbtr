"""java sample extraction: the `samples/java/` project, built as rbtr builds a repository and read as an agent reads it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_every_line_in_a_chunk, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "java") as build:
        symbols = outline(build)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("com/example/format/Formatter.java", "Formatter"): {
            ("com/example/shop/Order.java", "import com.example.format.Formatter;", "imports"),
            ("java.java", "Greeter", "inherits"),
            ("java.java", "import com.example.format.Formatter;", "imports"),
        },
        ("com/example/shop/Model.java", "Model"): {
            ("com/example/shop/Order.java", "Order", "inherits")
        },
        ("com/example/shop/Model.java", "check"): {
            ("com/example/shop/Model.java", "save", "calls"),
            ("com/example/shop/Order.java", "save", "calls"),
        },
        ("com/example/shop/Model.java", "save"): {("com/example/shop/Order.java", "save", "calls")},
        ("com/example/shop/Order.java", "Order"): {
            ("com/example/shop/Order.java", "build", "calls")
        },
        ("java.java", "Greeter"): {("java.java", "withDefault", "calls")},
        ("java.java", "greet"): {("java.java", "format", "calls")},
    }
    with indexed(Path(__file__).parent / "samples" / "java") as build:
        assert_reached(build, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "java") as build:
        assert_every_line_in_a_chunk(build)
