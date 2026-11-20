"""Java doc-comment extraction (Java is exterior-doc: leading comments attach via the sibling walk)."""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.testing import Indexed
from rbtr.testing.reads import read


@parametrize_with_cases("files, name, snippet", cases=".cases_docstrings", has_tag="documented")
def test_documented_chunk_includes_doc_text(
    indexed: Indexed, files: dict[str, str], name: str, snippet: str
) -> None:
    """By default the chunk content carries the symbol's docs."""
    with indexed(files) as index:
        [chunk] = read(index, name)
    assert snippet in chunk.content, f"{snippet!r} missing from {name}: {chunk.content!r}"


@parametrize_with_cases("files, name, snippet", cases=".cases_docstrings", has_tag="undocumented")
def test_no_phantom_documentation(
    indexed: Indexed, files: dict[str, str], name: str, snippet: str
) -> None:
    """Symbols without documentation do not gain any in content."""
    with indexed(files) as index:
        [chunk] = read(index, name)
    assert snippet not in chunk.content, f"{snippet!r} unexpected in {name}: {chunk.content!r}"
