"""Default docstring-extraction behaviour across languages.

Policy: rbtr extracts symbol-adjacent documentation into chunk
content by default for every supported language.  These tests
pin that policy down.

All test data lives in `case_docstrings.py`; each test is a
thin behavioural assertion over cases filtered by tag.  The
engine mechanism being probed is implicit in the tag slice, not
a branch inside any test function.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from tests.languages.conftest import extract_chunks


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases="tests.languages.case_docstrings",
    has_tag="documented",
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
    "lang, source, name, snippet",
    cases="tests.languages.case_docstrings",
    has_tag="documented",
)
def test_stripping_removes_doc_text(lang: str, source: str, name: str, snippet: str) -> None:
    """`--strip-docstrings` blanks the documentation text."""
    chunks = extract_chunks(lang, source, strip_docstrings=True)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet not in chunk.content, (
        f"expected {snippet!r} to be absent from stripped "
        f"{lang}.{name} content; got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases="tests.languages.case_docstrings",
    has_tag="documented",
)
def test_stripping_preserves_chunk_id(lang: str, source: str, name: str, snippet: str) -> None:
    """Stripping does not change `chunk.id`.

    `chunk_id` hashes `(file_path, name, line_start)`.  Stripping
    preserves `line_start` (docstring bytes are replaced with
    whitespace, not removed), so the ID must not shift.
    """
    kept = next(c for c in extract_chunks(lang, source) if c.name == name)
    stripped = next(
        c for c in extract_chunks(lang, source, strip_docstrings=True) if c.name == name
    )
    assert kept.id == stripped.id


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases="tests.languages.case_docstrings",
    has_tag="documented",
)
def test_stripping_preserves_line_count(lang: str, source: str, name: str, snippet: str) -> None:
    """Stripping preserves line_start, line_end, and the newline
    count in chunk content.
    """
    kept = next(c for c in extract_chunks(lang, source) if c.name == name)
    stripped = next(
        c for c in extract_chunks(lang, source, strip_docstrings=True) if c.name == name
    )
    assert kept.line_start == stripped.line_start
    assert kept.line_end == stripped.line_end
    assert kept.content.count("\n") == stripped.content.count("\n")


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases="tests.languages.case_docstrings",
    has_tag="undocumented",
)
def test_no_phantom_documentation(lang: str, source: str, name: str, snippet: str) -> None:
    """Symbols without documentation do not gain any in content.

    For `invalid` cases, the snippet marks text that *looks*
    like documentation but must not be swept into the chunk.
    """
    chunks = extract_chunks(lang, source)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet not in chunk.content, (
        f"unexpected {snippet!r} in {lang}.{name} content; got:\n{chunk.content!r}"
    )


# ── Engine-contract invariants over case slices ─────────────────────
#
# These two tests assert opposite outcomes about "what happens
# when leading-comment attachment is disabled".  The case set is
# partitioned by the `interior_doc` / `exterior_doc` tags so
# each test runs against only the language family it applies to
# — no `if/else` branching inside the test function.


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases="tests.languages.case_docstrings",
    has_tag="exterior_doc",
)
def test_no_leading_attachment_drops_exterior_docs(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """Forcing `doc_comment_node_types=frozenset()` disables the
    sibling walk.  For exterior-doc plugins the snippet comes
    from a leading comment, which is now absent.
    """
    chunks = extract_chunks(lang, source, no_leading_attachment=True)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet not in chunk.content, (
        f"expected {snippet!r} to be absent from {lang}.{name} "
        f"when leading-comment attachment is suppressed; "
        f"got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases="tests.languages.case_docstrings",
    has_tag="interior_doc",
)
def test_no_leading_attachment_preserves_interior_docs(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """Interior-doc plugins (Python) use the `@_docstring` query
    capture, which is orthogonal to leading-comment attachment.
    Forcing the override has no effect.
    """
    chunks = extract_chunks(lang, source, no_leading_attachment=True)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet in chunk.content


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases="tests.languages.case_docstrings",
    has_tag="exterior_doc",
)
def test_attachment_shifts_line_start_for_exterior_docs(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """When sibling-walk attachment fires, `line_start` moves
    up to cover the earliest attached comment.  Compared
    against the override-empty extraction: exterior plugins
    shift, interior plugins do not.
    """
    default = next(c for c in extract_chunks(lang, source) if c.name == name)
    no_attach = next(
        c for c in extract_chunks(lang, source, no_leading_attachment=True) if c.name == name
    )
    assert default.line_start < no_attach.line_start
    assert default.id != no_attach.id


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases="tests.languages.case_docstrings",
    has_tag="interior_doc",
)
def test_attachment_does_not_shift_line_start_for_interior_docs(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """Interior-doc plugins (Python) do not run the sibling
    walk; `line_start` is identical with or without the
    override.
    """
    default = next(c for c in extract_chunks(lang, source) if c.name == name)
    no_attach = next(
        c for c in extract_chunks(lang, source, no_leading_attachment=True) if c.name == name
    )
    assert default.line_start == no_attach.line_start
    assert default.id == no_attach.id
