"""Tests for prose and fallback chunking."""

from rbtr.index.chunks import chunk_markdown, chunk_plaintext
from rbtr.index.models import ChunkKind


def test_markdown_splits_by_heading() -> None:
    content = """\
# Title

Intro text.

## Section A

Body A.

## Section B

Body B.
"""
    chunks = chunk_markdown("doc.md", "sha1", content)
    names = [c.name for c in chunks]
    assert "Title" in names
    assert "Section A" in names
    assert "Section B" in names


def test_markdown_preserves_scope_chain() -> None:
    content = """\
# Top

## Mid

### Deep

Content here.
"""
    chunks = chunk_markdown("doc.md", "sha1", content)
    deep = next(c for c in chunks if c.name == "Deep")
    assert deep.scope == "Top > Mid"


def test_markdown_all_doc_section_kind() -> None:
    content = """\
# Heading

Some text.
"""
    chunks = chunk_markdown("doc.md", "sha1", content)
    assert all(c.kind == ChunkKind.DOC_SECTION for c in chunks)


def test_markdown_sets_line_numbers() -> None:
    content = """\
# First

Text.

## Second

More text.
"""
    chunks = chunk_markdown("doc.md", "sha1", content)
    first = next(c for c in chunks if c.name == "First")
    second = next(c for c in chunks if c.name == "Second")
    assert first.line_start == 1
    assert second.line_start == 5


def test_markdown_no_headings_falls_back_to_raw() -> None:
    content = """\
Just some plain text
without any headings.
"""
    chunks = chunk_markdown("doc.md", "sha1", content)
    assert len(chunks) >= 1
    assert all(c.kind == ChunkKind.RAW_CHUNK for c in chunks)


def test_plaintext_chunks_small_file() -> None:
    content = "\n".join(f"line {i}" for i in range(10))
    chunks = chunk_plaintext("file.txt", "sha1", content)
    assert len(chunks) == 1
    assert chunks[0].kind == ChunkKind.RAW_CHUNK


def test_plaintext_chunks_large_file() -> None:
    content = "\n".join(f"line {i}" for i in range(120))
    chunks = chunk_plaintext("file.txt", "sha1", content)
    # 120 lines with chunk size 50 and overlap 5 → multiple chunks.
    assert len(chunks) > 1


def test_markdown_sets_blob_sha() -> None:
    content = """\
# Heading

Body.
"""
    chunks = chunk_markdown("doc.md", "abc123", content)
    assert all(c.blob_sha == "abc123" for c in chunks)


def test_markdown_empty_content() -> None:
    chunks = chunk_markdown("empty.md", "sha1", "")
    assert chunks == []
