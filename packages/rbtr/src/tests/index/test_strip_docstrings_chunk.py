"""Tests for the per-chunk `strip_docstrings` column."""

from __future__ import annotations

import pytest

from rbtr.index.models import Chunk, ChunkKind, IndexVariant
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code
from rbtr.index.treesitter import _chunk_id


@pytest.fixture
def full_chunk() -> Chunk:
    name = "load_config"
    file_path = "src/config.py"
    line_start = 1
    content = 'def load_config():\n    """Load."""\n    pass\n'
    return Chunk(
        id=_chunk_id(file_path, name, line_start, strip_docstrings=False),
        blob_sha="blob_test",
        file_path=file_path,
        kind=ChunkKind.FUNCTION,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=line_start,
        line_end=line_start + 2,
        strip_docstrings=False,
    )


@pytest.fixture
def stripped_chunk() -> Chunk:
    name = "load_config"
    file_path = "src/config.py"
    line_start = 1
    content = "def load_config():\n    \n    pass\n"
    return Chunk(
        id=_chunk_id(file_path, name, line_start, strip_docstrings=True),
        blob_sha="blob_test",
        file_path=file_path,
        kind=ChunkKind.FUNCTION,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=line_start,
        line_end=line_start + 2,
        strip_docstrings=True,
    )


@pytest.fixture
def populated_store(full_chunk: Chunk, stripped_chunk: Chunk) -> IndexStore:
    """In-memory store with both chunk variants and one snapshot."""
    store = IndexStore()
    repo_id = store.register_repo("/test/repo")
    store.insert_chunks([full_chunk, stripped_chunk], repo_id=repo_id)
    store.insert_snapshot(
        "HEAD", full_chunk.file_path, full_chunk.blob_sha, repo_id=repo_id
    )
    return store


def test_chunk_id_differs_by_variant(full_chunk: Chunk, stripped_chunk: Chunk) -> None:
    assert full_chunk.id != stripped_chunk.id


def test_full_variant_returns_only_full_chunk(
    populated_store: IndexStore, full_chunk: Chunk, stripped_chunk: Chunk
) -> None:
    rows = populated_store.get_chunks("HEAD", variant=IndexVariant.FULL)
    ids = {c.id for c in rows}
    assert full_chunk.id in ids
    assert stripped_chunk.id not in ids


def test_stripped_variant_returns_only_stripped_chunk(
    populated_store: IndexStore, full_chunk: Chunk, stripped_chunk: Chunk
) -> None:
    rows = populated_store.get_chunks("HEAD", variant=IndexVariant.STRIPPED)
    ids = {c.id for c in rows}
    assert stripped_chunk.id in ids
    assert full_chunk.id not in ids


def test_full_chunk_preserves_docstring(
    populated_store: IndexStore, full_chunk: Chunk
) -> None:
    [row] = populated_store.get_chunks("HEAD", variant=IndexVariant.FULL)
    assert row.strip_docstrings is False
    assert "Load" in row.content


def test_stripped_chunk_blanks_docstring(
    populated_store: IndexStore, stripped_chunk: Chunk
) -> None:
    [row] = populated_store.get_chunks("HEAD", variant=IndexVariant.STRIPPED)
    assert row.strip_docstrings is True
    assert "Load" not in row.content


def test_strip_docstrings_defaults_to_false() -> None:
    chunk = Chunk(
        id="default_test",
        blob_sha="b",
        file_path="src/x.py",
        kind=ChunkKind.FUNCTION,
        name="x",
        content="def x(): pass",
        line_start=1,
        line_end=1,
    )
    assert chunk.strip_docstrings is False
