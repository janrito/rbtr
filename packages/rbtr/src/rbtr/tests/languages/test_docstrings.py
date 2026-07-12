"""Engine-level doc-attachment tests.

The engine folds symbol-adjacent documentation into chunk content two ways:
an *interior* docstring (Python's `@_docstring` query capture) and an
*exterior* leading comment (the sibling walk driven by
`doc_comment_node_types`). Per-language doc behaviour is tested in each
package; these tests pin the engine mechanism itself using two default
languages — C (exterior) and Python (interior) — including the `line_start`
shift that only exterior attachment causes.
"""

from __future__ import annotations

from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file, extract_query


def test_suppressing_attachment_drops_exterior_doc() -> None:
    """Disabling the sibling walk drops an exterior leading-comment doc (C)."""
    src = b"// Compute the answer.\nint answer(void) { return 42; }\n"
    chunk = next(
        c
        for c in extract_query("c", "x.c", "sha1", src, doc_comment_node_types=frozenset())
        if c.name == "answer"
    )
    assert "Compute" not in chunk.content


def test_attachment_shifts_line_start_for_exterior_doc() -> None:
    """Attaching an exterior comment moves `line_start` up to cover it (C)."""
    src = b"// Compute the answer.\nint answer(void) { return 42; }\n"
    default = next(
        c for c in extract_file(FileEntry("x.c", "sha1", src), "c") if c.name == "answer"
    )
    suppressed = next(
        c
        for c in extract_query("c", "x.c", "sha1", src, doc_comment_node_types=frozenset())
        if c.name == "answer"
    )
    assert default.line_start < suppressed.line_start
    assert default.id != suppressed.id


def test_suppressing_attachment_preserves_interior_doc() -> None:
    """Interior docstrings (Python's `@_docstring`) are orthogonal to the walk override."""
    src = b'def answer():\n    """Compute the answer."""\n    return 42\n'
    chunk = next(
        c
        for c in extract_query("python", "x.py", "sha1", src, doc_comment_node_types=frozenset())
        if c.name == "answer"
    )
    assert "Compute" in chunk.content


def test_attachment_does_not_shift_line_start_for_interior_doc() -> None:
    """Interior docs don't run the sibling walk, so `line_start` is unchanged (Python)."""
    src = b'def answer():\n    """Compute the answer."""\n    return 42\n'
    default = next(
        c for c in extract_file(FileEntry("x.py", "sha1", src), "python") if c.name == "answer"
    )
    suppressed = next(
        c
        for c in extract_query("python", "x.py", "sha1", src, doc_comment_node_types=frozenset())
        if c.name == "answer"
    )
    assert default.line_start == suppressed.line_start
    assert default.id == suppressed.id
