"""TOML extraction tests (cases in `cases_extraction.py`)."""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file


@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


def test_toml_dotted_key_scope() -> None:
    """A TOML dotted table splits into last-segment name + preceding scope."""
    src = """\
[tool.ruff.lint]
select = ["E"]
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "toml")
    assert chunks[0].name == "lint"
    assert chunks[0].scope == "tool::ruff"
