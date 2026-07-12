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

from rbtr.git import FileEntry
from rbtr.index.orchestrator import extract_file, extract_query


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=".cases_docstrings",
    has_tag="documented",
)
def test_documented_chunk_includes_doc_text(
    lang: str,
    source: str,
    name: str,
    snippet: str,
) -> None:
    """By default the chunk content carries the symbol's docs."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    chunk = next(c for c in chunks if c.name == name)
    assert snippet in chunk.content, (
        f"expected {snippet!r} in {lang}.{name} content; got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=".cases_docstrings",
    has_tag="undocumented",
)
def test_no_phantom_documentation(lang: str, source: str, name: str, snippet: str) -> None:
    """Symbols without documentation do not gain any in content.

    For `invalid` cases, the snippet marks text that *looks*
    like documentation but must not be swept into the chunk.
    """
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
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
    cases=".cases_docstrings",
    has_tag="exterior_doc",
)
def test_no_leading_attachment_drops_exterior_docs(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """Forcing `doc_comment_node_types=frozenset()` disables the
    sibling walk.  For exterior-doc plugins the snippet comes
    from a leading comment, which is now absent.
    """
    chunks = list(
        extract_query(lang, "input", "sha1", source.encode(), doc_comment_node_types=frozenset())
    )
    chunk = next(c for c in chunks if c.name == name)
    assert snippet not in chunk.content, (
        f"expected {snippet!r} to be absent from {lang}.{name} "
        f"when leading-comment attachment is suppressed; "
        f"got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=".cases_docstrings",
    has_tag="interior_doc",
)
def test_no_leading_attachment_preserves_interior_docs(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """Interior-doc plugins (Python) use the `@_docstring` query
    capture, which is orthogonal to leading-comment attachment.
    Forcing the override has no effect.
    """
    chunks = list(
        extract_query(lang, "input", "sha1", source.encode(), doc_comment_node_types=frozenset())
    )
    chunk = next(c for c in chunks if c.name == name)
    assert snippet in chunk.content


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=".cases_docstrings",
    has_tag="exterior_doc",
)
def test_attachment_shifts_line_start_for_exterior_docs(
    lang: str,
    source: str,
    name: str,
    snippet: str,
) -> None:
    """When sibling-walk attachment fires, `line_start` moves
    up to cover the earliest attached comment.  Compared
    against the override-empty extraction: exterior plugins
    shift, interior plugins do not.
    """
    default = next(
        c for c in extract_file(FileEntry("input", "sha1", source.encode()), lang) if c.name == name
    )
    suppressed = extract_query(
        lang, "input", "sha1", source.encode(), doc_comment_node_types=frozenset()
    )
    no_attach = next(c for c in suppressed if c.name == name)
    assert default.line_start < no_attach.line_start
    assert default.id != no_attach.id


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=".cases_docstrings",
    has_tag="interior_doc",
)
def test_attachment_does_not_shift_line_start_for_interior_docs(
    lang: str,
    source: str,
    name: str,
    snippet: str,
) -> None:
    """Interior-doc plugins (Python) do not run the sibling
    walk; `line_start` is identical with or without the
    override.
    """
    default = next(
        c for c in extract_file(FileEntry("input", "sha1", source.encode()), lang) if c.name == name
    )
    suppressed = extract_query(
        lang, "input", "sha1", source.encode(), doc_comment_node_types=frozenset()
    )
    no_attach = next(c for c in suppressed if c.name == name)
    assert default.line_start == no_attach.line_start
    assert default.id == no_attach.id
