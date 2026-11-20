"""CSS extraction tests.

Symbol cases (`cases_extraction.py`) drive the shared check; the function
at the end pins CSS's `@import` edge behaviour.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.domain.models import ChunkKind
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


def test_css_import_produces_import_chunk(indexed: Indexed) -> None:
    """CSS @import url(...) produces an import chunk."""
    src = """\
@import url("reset.css");
body { color: #333; }
"""
    with indexed({"input.css": src}) as index:
        chunks = outline(index)
    imports = [c.metadata for c in chunks if c.kind == ChunkKind.IMPORT and c.metadata]
    assert len(imports) == 1
    assert imports[0].module == "reset.css"
