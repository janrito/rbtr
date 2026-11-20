"""C++ sample extraction: the `samples/cpp/` project, built as rbtr builds a repository and read as an agent reads it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_every_line_in_a_chunk, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "cpp") as build:
        symbols = outline(build)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("greeter.hpp", "Greeting"): {("cpp.cpp", '#include "greeter.hpp"', "imports")},
        ("greeter.hpp", "greet"): {("cpp.cpp", '#include "greeter.hpp"', "imports")},
        ("greeter.hpp", "render"): {("cpp.cpp", '#include "greeter.hpp"', "imports")},
        ("greeter.hpp", "withDefault"): {("cpp.cpp", '#include "greeter.hpp"', "imports")},
        ("shop/model.cpp", "check"): {
            ("shop/model.cpp", "save", "calls"),
            ("shop/order.cpp", "save", "calls"),
        },
        ("shop/model.cpp", "save"): {("shop/order.cpp", "save", "calls")},
        ("shop/model.hpp", "Model"): {
            ("shop/model.cpp", '#include "model.hpp"', "imports"),
            ("shop/order.hpp", '#include "model.hpp"', "imports"),
            ("shop/order.hpp", "Order", "inherits"),
        },
        ("shop/model.hpp", "check"): {
            ("shop/model.cpp", '#include "model.hpp"', "imports"),
            ("shop/order.cpp", "save", "calls"),
            ("shop/order.hpp", '#include "model.hpp"', "imports"),
        },
        ("shop/model.hpp", "save"): {
            ("shop/model.cpp", '#include "model.hpp"', "imports"),
            ("shop/order.cpp", "save", "calls"),
            ("shop/order.hpp", '#include "model.hpp"', "imports"),
        },
        ("shop/model.hpp", "shop"): {
            ("shop/model.cpp", '#include "model.hpp"', "imports"),
            ("shop/order.hpp", '#include "model.hpp"', "imports"),
        },
        ("shop/order.hpp", "Order"): {("shop/order.cpp", '#include "order.hpp"', "imports")},
        ("shop/order.hpp", "save"): {("shop/order.cpp", '#include "order.hpp"', "imports")},
        ("shop/order.hpp", "shop"): {("shop/order.cpp", '#include "order.hpp"', "imports")},
        ("util/text.hpp", "text_length"): {("util/text.cpp", '#include "text.hpp"', "imports")},
        ("util/text.hpp", "util"): {("util/text.cpp", '#include "text.hpp"', "imports")},
    }
    with indexed(Path(__file__).parent / "samples" / "cpp") as build:
        assert_reached(build, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "cpp") as build:
        assert_every_line_in_a_chunk(build)
