"""C sample extraction: the `samples/c/` project, built as rbtr builds a repository and read as an agent reads it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_every_line_in_a_chunk, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "c") as build:
        symbols = outline(build)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("greeter.h", "GREETER_H"): {("c.c", '#include "greeter.h"', "imports")},
        ("greeter.h", "Greeter"): {("c.c", '#include "greeter.h"', "imports")},
        ("greeter.h", "MAX_NAME"): {("c.c", '#include "greeter.h"', "imports")},
        ("greeter.h", "format_greeting"): {("c.c", '#include "greeter.h"', "imports")},
        ("greeter.h", "greeter_default"): {("c.c", '#include "greeter.h"', "imports")},
        ("shop/shop.c", "shop_total"): {("shop/use.c", "use_shop", "calls")},
        ("shop/shop.c", "valid"): {("shop/shop.c", "shop_total", "calls")},
        ("shop/shop.h", "shop_total"): {
            ("shop/shop.c", '#include "shop.h"', "imports"),
            ("shop/use.c", '#include "shop.h"', "imports"),
            ("shop/use.c", "use_shop", "calls"),
        },
        ("util/text.h", "GREETER_UTIL_TEXT_H"): {("util/text.c", '#include "text.h"', "imports")},
        ("util/text.h", "text_length"): {("util/text.c", '#include "text.h"', "imports")},
    }
    with indexed(Path(__file__).parent / "samples" / "c") as build:
        assert_reached(build, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "c") as build:
        assert_every_line_in_a_chunk(build)
