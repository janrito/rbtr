"""Engine-level doc-attachment tests.

The engine folds symbol-adjacent documentation into chunk content two ways:
an *interior* docstring (Python's `@_docstring` query capture) and an
*exterior* leading comment (a `@comment` block folded into the symbol flush
after it). Per-language doc behaviour is tested in each package; these tests
pin the engine mechanism itself using two default languages — C (exterior)
and Python (interior) — including the `line_start` shift that only exterior
attachment causes, and that a blank-separated comment stays standalone.
"""

from __future__ import annotations

from rbtr.git import FileEntry
from rbtr.index.models import ChunkKind
from rbtr.languages.extract import extract_file


def test_exterior_leading_comment_folds_into_symbol() -> None:
    """A `@comment` block flush before a symbol folds into it, covering L1 (C)."""
    src = b"// Compute the answer.\nint answer(void) { return 42; }\n"
    chunk = next(c for c in extract_file(FileEntry("x.c", "sha1", src), "c") if c.name == "answer")
    assert "Compute" in chunk.content
    assert chunk.line_start == 1


def test_blank_separated_comment_stays_standalone() -> None:
    """A comment separated by a blank line is a standalone COMMENT, not folded (C)."""
    src = b"// A banner, attached to nothing.\n\nint answer(void) { return 42; }\n"
    chunks = extract_file(FileEntry("x.c", "sha1", src), "c")
    comments = [c for c in chunks if c.kind == ChunkKind.COMMENT]
    assert len(comments) == 1
    assert "banner" in comments[0].content
    answer = next(c for c in chunks if c.name == "answer")
    assert "banner" not in answer.content


def test_interior_docstring_is_in_content_without_shifting_line_start() -> None:
    """A Python interior docstring is part of the function chunk and, unlike an
    exterior leading comment, does not move `line_start` above the def."""
    src = b'def answer():\n    """Compute the answer."""\n    return 42\n'
    chunk = next(
        c for c in extract_file(FileEntry("x.py", "sha1", src), "python") if c.name == "answer"
    )
    assert "Compute" in chunk.content
    assert chunk.line_start == 1
