"""CSS extraction tests.

Symbol cases (`cases_extraction.py`) drive the shared check; the function
at the end pins CSS's `@import` edge behaviour.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.index.models import ChunkKind
from rbtr.languages.extract import extract_file


@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


def test_css_import_produces_import_chunk() -> None:
    """CSS @import url(...) produces an import chunk."""
    src = """\
@import url("reset.css");
body { color: #333; }
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "css")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "reset.css"
