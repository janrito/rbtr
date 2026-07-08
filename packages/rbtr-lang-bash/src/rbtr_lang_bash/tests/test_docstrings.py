"""Bash docstring-extraction tests.

Bash docs are exterior (a leading `#` comment run); there are no
interior-doc cases, so only the documented / undocumented / exterior-doc
checks apply.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.languages.testkit import extract_chunks


@parametrize_with_cases(
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="documented"
)
def test_documented_chunk_includes_doc_text(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """By default the chunk content carries the symbol's docs."""
    chunks = extract_chunks(lang, source)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet in chunk.content, (
        f"expected {snippet!r} in {lang}.{name} content; got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="undocumented"
)
def test_no_phantom_documentation(lang: str, source: str, name: str, snippet: str) -> None:
    """Symbols without documentation do not gain any in content."""
    chunks = extract_chunks(lang, source)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet not in chunk.content, (
        f"unexpected {snippet!r} in {lang}.{name} content; got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="exterior_doc"
)
def test_no_leading_attachment_drops_exterior_docs(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """Forcing `doc_comment_node_types=frozenset()` disables the sibling walk,
    so the leading-comment snippet is now absent."""
    chunks = extract_chunks(lang, source, no_leading_attachment=True)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet not in chunk.content, (
        f"expected {snippet!r} absent from {lang}.{name} when leading-comment "
        f"attachment is suppressed; got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet", cases=".cases_docstrings", has_tag="exterior_doc"
)
def test_attachment_shifts_line_start_for_exterior_docs(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """When sibling-walk attachment fires, `line_start` moves up to cover the
    earliest attached comment; suppressing it shifts the start back."""
    default = next(c for c in extract_chunks(lang, source) if c.name == name)
    no_attach = next(
        c for c in extract_chunks(lang, source, no_leading_attachment=True) if c.name == name
    )
    assert default.line_start < no_attach.line_start
    assert default.id != no_attach.id
