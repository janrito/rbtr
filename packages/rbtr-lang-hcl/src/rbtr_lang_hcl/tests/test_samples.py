"""HCL sample extraction: the `samples/hcl/` project, built as rbtr builds a repository and read as an agent reads it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_every_line_in_a_chunk, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "hcl") as build:
        symbols = outline(build)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    # A module block is an import named by its whole text.
    greeter = """\
module "greeter" {
  source = "./modules/greeter"
}"""
    module_x = """\
module "x" {
  source = "./b"
}"""
    reached = {
        ("modules/greeter/main.tf", "output greeting"): {("main.tf", greeter, "imports")},
        ("modules/greeter/main.tf", "variable name"): {("main.tf", greeter, "imports")},
        ("modules/greeter/versions.tf", "terraform"): {("main.tf", greeter, "imports")},
        ("tf/b/main.tf", "resource null_resource y"): {("tf/a.tf", module_x, "imports")},
    }
    with indexed(Path(__file__).parent / "samples" / "hcl") as build:
        assert_reached(build, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "hcl") as build:
        assert_every_line_in_a_chunk(build)
