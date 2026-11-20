"""JSON extraction tests (cases in `cases_extraction.py`)."""

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


def test_a_ref_key_is_one_symbol(indexed: Indexed) -> None:
    """A `$ref` is a key that also imports the file it names: one symbol, which reaches the key its pointer names."""
    files = {
        "schema.json": '{"$ref": "defs.json#/Name"}\n',
        "defs.json": '{"Name": {}}\n',
    }
    with indexed(files) as build:
        reply = outline(build, ["schema.json"])
        assert [(s.kind, s.name, s.line_start, s.line_end) for s in reply] == [
            ("config_key", "$ref", 1, 1)
        ]
        assert_reached(build, {("defs.json", "Name"): {("schema.json", "$ref", "imports")}})
