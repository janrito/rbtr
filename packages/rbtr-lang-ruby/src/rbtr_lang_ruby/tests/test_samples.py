"""Ruby sample extraction: the `samples/ruby/` project, built as rbtr builds a repository and read as an agent reads it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "ruby") as build:
        symbols = outline(build)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("config.rb", "DEFAULT_LOCALE"): {("ruby.rb", 'require_relative "./config"', "imports")},
        ("lib/b.rb", "helper"): {("lib/a.rb", "require_relative 'b'", "imports")},
        ("lib/model.rb", "Model"): {
            ("lib/order.rb", "Order", "inherits"),
            ("lib/order.rb", 'require_relative "model"', "imports"),
        },
        ("lib/model.rb", "check"): {
            ("lib/model.rb", "save", "calls"),
            ("lib/order.rb", 'require_relative "model"', "imports"),
            ("lib/order.rb", "save", "calls"),
        },
        ("lib/model.rb", "save"): {("lib/order.rb", 'require_relative "model"', "imports")},
        ("lib/order.rb", "Order"): {("lib/order.rb", "build", "calls")},
        ("ruby.rb", "Greeter"): {
            ("ruby.rb", "format_greeting", "calls"),
            ("ruby.rb", "includes the configured prefix", "calls"),
        },
        ("ruby.rb", "Shoutable"): {("ruby.rb", "Greeter", "inherits")},
    }
    with indexed(Path(__file__).parent / "samples" / "ruby") as build:
        assert_reached(build, reached)
