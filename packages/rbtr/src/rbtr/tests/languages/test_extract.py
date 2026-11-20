"""Tests for routing a file to its language, through the engine."""

from __future__ import annotations

import pytest

from rbtr.domain.models import ChunkKind
from rbtr.testing import Indexed
from rbtr.testing.reads import outline


@pytest.mark.parametrize(
    ("file_path", "file_content", "expected"),
    [
        pytest.param(
            "mod.py",
            """\
def greet():
    return 1
""",
            [("python", ChunkKind.FUNCTION, "greet")],
            id="python",
        ),
        pytest.param(
            "doc.md",
            """\
# Title

Body text.
""",
            [("markdown", ChunkKind.DOC_SECTION, "Title")],
            id="markdown",
        ),
        pytest.param(
            "doc.rst",
            """\
Title
=====

Body text.
""",
            [("rst", ChunkKind.DOC_SECTION, "Title")],
            id="rst",
        ),
        pytest.param(
            "CHANGES.txt",
            """\
Title
=====

Body text.
""",
            [("rst", ChunkKind.DOC_SECTION, "Title")],
            id="unknown-ext-rst",
        ),
        pytest.param(
            "README.txt",
            """\
# Title

Body text.
""",
            [("markdown", ChunkKind.DOC_SECTION, "Title")],
            id="unknown-ext-md",
        ),
        pytest.param(
            "empty.txt",
            "",
            [("plaintext", ChunkKind.RAW_CHUNK, "")],
            id="empty-file",
        ),
        pytest.param(
            "blank.txt",
            "   \n\n  \n",
            [("plaintext", ChunkKind.RAW_CHUNK, "")],
            id="whitespace-only",
        ),
        pytest.param(
            "data.txt",
            """\
Just some plain text.
Nothing special.
""",
            [("plaintext", ChunkKind.RAW_CHUNK, "")],
            id="unknown-ext-plain",
        ),
    ],
)
def test_a_file_is_extracted_by_the_language_it_is(
    indexed: Indexed,
    file_path: str,
    file_content: str,
    expected: list[tuple[str, ChunkKind, str]],
) -> None:
    """With every language installed, a file's path or text picks the language that extracts it."""
    with indexed({file_path: file_content}) as index:
        symbols = outline(index, [file_path])
    assert [(s.language, s.kind, s.name) for s in symbols] == expected
