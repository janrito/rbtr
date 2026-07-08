"""JavaScript / TypeScript JSDoc extraction.

rbtr folds a symbol's leading JSDoc into its chunk content. JS/TS are
exterior-doc languages (the doc is a leading comment attached by the
sibling walk), so suppressing that walk drops the doc and shifts
`line_start`. Data lives in `cases_docstrings.py`, sliced by tag.
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
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="exterior_doc"
)
def test_leading_doc_folds_into_symbol(lang: str, source: str, name: str, snippet: str) -> None:
    """A leading comment block folds into its symbol's chunk content."""
    chunk = next(
        c for c in extract_file(FileEntry("input", "sha1", source.encode()), lang) if c.name == name
    )
    assert snippet in chunk.content
