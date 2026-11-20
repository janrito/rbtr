"""TOML extraction tests (cases in `cases_extraction.py`)."""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.testing import Indexed
from rbtr.testing.reads import outline


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(indexed: Indexed, files: dict[str, str], expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the outline."""
    with indexed(files) as index:
        reply = outline(index)
    symbols = [(s.kind, s.name, s.scope) for s in reply]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


def test_toml_dotted_key_scope(indexed: Indexed) -> None:
    """A TOML dotted table splits into last-segment name + preceding scope."""
    src = """\
[tool.ruff.lint]
select = ["E"]
"""
    with indexed({"input.toml": src}) as index:
        chunks = outline(index)
    assert chunks[0].name == "lint"
    assert chunks[0].scope == "tool::ruff"
