"""Tests for per-file extraction routing (`extract_file`)."""

from __future__ import annotations

import pytest

from rbtr.git import FileEntry
from rbtr.index.models import ChunkKind
from rbtr.languages.extract import extract_file


@pytest.mark.parametrize(
    (
        "file_path",
        "file_content",
        "expected_language",
        "expected_kind",
        "expected_name",
        "min_chunks",
    ),
    [
        pytest.param(
            "pkg.json",
            '{"name": "test", "version": "1.0"}',
            "json",
            ChunkKind.DOC_SECTION,
            None,
            1,
            id="json",
        ),
        pytest.param(
            "doc.md",
            """\
# Title

Body text.
""",
            "markdown",
            None,
            "Title",
            1,
            id="markdown",
        ),
        pytest.param(
            "doc.rst",
            """\
Title
=====

Body text.
""",
            "rst",
            None,
            "Title",
            1,
            id="rst",
        ),
        pytest.param(
            "config.toml",
            """\
[project]
name = 'rbtr'
""",
            "toml",
            None,
            "project",
            1,
            id="toml",
        ),
        pytest.param(
            "ci.yml",
            """\
name: CI
on: [push]
""",
            "yaml",
            None,
            None,
            1,
            id="yaml",
        ),
        pytest.param(
            "CHANGES.txt",
            """\
Title
=====

Body text.
""",
            "rst",
            None,
            None,
            1,
            id="unknown-ext-rst",
        ),
        pytest.param(
            "README.txt",
            """\
# Title

Body text.
""",
            "markdown",
            None,
            None,
            1,
            id="unknown-ext-md",
        ),
        pytest.param(
            "empty.txt",
            "",
            "",
            None,
            None,
            0,
            id="empty-file",
        ),
        pytest.param(
            "blank.txt",
            "   \n\n  \n",
            "",
            None,
            None,
            0,
            id="whitespace-only",
        ),
        pytest.param(
            "data.txt",
            """\
Just some plain text.
Nothing special.
""",
            "",
            ChunkKind.RAW_CHUNK,
            None,
            1,
            id="unknown-ext-plain",
        ),
    ],
)
def test_extract_file_routes_correctly(
    file_path: str,
    file_content: str,
    expected_language: str,
    expected_kind: ChunkKind | None,
    expected_name: str | None,
    min_chunks: int,
) -> None:
    """extract_file routes each file type to the correct chunker."""
    entry = FileEntry(path=file_path, blob_sha="sha1", content=file_content.encode())
    chunks = list(extract_file(entry, expected_language))
    assert len(chunks) >= min_chunks
    if not chunks:
        return
    assert chunks[0].language == expected_language
    if expected_kind is not None:
        assert all(c.kind == expected_kind for c in chunks)
    if expected_name is not None:
        assert chunks[0].name == expected_name
