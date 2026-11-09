"""Rust doc-comment extraction (Rust is exterior-doc: leading comments attach via the sibling walk)."""

from __future__ import annotations

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.languages.extract import docstring, extract_file


@parametrize_with_cases(
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="documented"
)
def test_documented_chunk_includes_doc_text(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """By default the chunk content carries the symbol's docs."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet in chunk.content, f"expected {snippet!r} in {lang}.{name}: {chunk.content!r}"


@parametrize_with_cases(
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="undocumented"
)
def test_no_phantom_documentation(lang: str, source: str, name: str, snippet: str) -> None:
    """Symbols without documentation do not gain any in content."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet not in chunk.content, (
        f"unexpected {snippet!r} in {lang}.{name}: {chunk.content!r}"
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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            """\
/// Return a friendly greeting.
fn greet(name: &str) -> String {
    name.to_string()
}
""",
            "/// Return a friendly greeting.\n",
        ),
        (
            """\
fn greet(name: &str) -> String {
    name.to_string()
}
""",
            None,
        ),
    ],
    ids=["doc-comment", "undocumented"],
)
def test_docstring_is_the_doc_comment_as_written(source: str, expected: str | None) -> None:
    """`docstring` returns the first documented symbol's doc comment as written, or none."""
    assert docstring(source, "rust") == expected
