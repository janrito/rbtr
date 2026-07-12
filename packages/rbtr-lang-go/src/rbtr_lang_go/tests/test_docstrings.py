"""Go doc-comment extraction (Go is exterior-doc: leading comments attach via the sibling walk)."""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file, extract_query


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
def test_no_leading_attachment_drops_exterior_docs(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """Suppressing the sibling walk drops the leading-comment doc."""
    chunks = list(
        extract_query(lang, "input", "sha1", source.encode(), doc_comment_node_types=frozenset())
    )
    chunk = next(c for c in chunks if c.name == name)
    assert snippet not in chunk.content


@parametrize_with_cases(
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="exterior_doc"
)
def test_attachment_shifts_line_start_for_exterior_docs(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """Attachment moves line_start up to the earliest attached comment."""
    default = next(
        c for c in extract_file(FileEntry("input", "sha1", source.encode()), lang) if c.name == name
    )
    suppressed = extract_query(
        lang, "input", "sha1", source.encode(), doc_comment_node_types=frozenset()
    )
    no_attach = next(c for c in suppressed if c.name == name)
    assert default.line_start < no_attach.line_start
    assert default.id != no_attach.id
