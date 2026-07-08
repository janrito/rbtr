"""Python docstring extraction.

rbtr folds a symbol's docstring into its chunk content. Python is an
interior-doc language — the docstring is captured by the `@_docstring` query
capture, orthogonal to the leading-comment sibling walk — so suppressing that
walk leaves the doc (and `line_start`) unchanged. Data lives in
`cases_docstrings.py`, sliced by tag.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file


@parametrize_with_cases(
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="documented"
)
def test_documented_chunk_includes_doc_text(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """By default the chunk content carries the symbol's docs."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet in chunk.content, (
        f"expected {snippet!r} in {lang}.{name} content; got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="undocumented"
)
def test_no_phantom_documentation(lang: str, source: str, name: str, snippet: str) -> None:
    """Symbols without documentation do not gain any in content."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet not in chunk.content, (
        f"unexpected {snippet!r} in {lang}.{name} content; got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="interior_doc"
)
def test_interior_docstring_folds_into_content(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """An interior (`@_docstring`) doc is part of its symbol's chunk content."""
    chunk = next(
        c for c in extract_file(FileEntry("input", "sha1", source.encode()), lang) if c.name == name
    )
    assert snippet in chunk.content
