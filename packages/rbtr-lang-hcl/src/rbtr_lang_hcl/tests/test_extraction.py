"""HCL extraction tests (cases in `cases_extraction.py`)."""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_reached
from rbtr.testing.reads import outline


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(indexed: Indexed, files: dict[str, str], expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the outline."""
    with indexed(files) as build:
        reply = outline(build)
    symbols = [(s.kind, s.name, s.scope) for s in reply]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


def test_a_module_block_is_one_symbol(indexed: Indexed) -> None:
    """A `module` block is a key that also imports its `source`: one symbol, which reaches the module."""
    files = {
        "main.tf": """\
module "x" {
  source = "./b"
}
""",
        "b/main.tf": """\
variable "region" {}
""",
    }
    with indexed(files) as build:
        reply = outline(build, ["main.tf"])
        assert [(s.kind, s.name, s.line_start, s.line_end) for s in reply] == [
            ("config_key", "module x", 1, 3)
        ]
        assert_reached(
            build, {("b/main.tf", "variable region"): {("main.tf", "module x", "imports")}}
        )
