"""Tests for the per-chunk `strip_docstrings` column."""

from __future__ import annotations

from rbtr.index.models import Chunk, ChunkKind, IndexVariant
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code
from rbtr.index.treesitter import _chunk_id


def _make_chunk(*, content: str, strip_docstrings: bool) -> Chunk:
    """Build a Chunk for the same logical symbol in either variant."""
    name = "load_config"
    file_path = "src/config.py"
    line_start = 1
    return Chunk(
        id=_chunk_id(file_path, name, line_start, strip_docstrings),
        blob_sha="blob_test",
        file_path=file_path,
        kind=ChunkKind.FUNCTION,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=line_start,
        line_end=line_start + 2,
        strip_docstrings=strip_docstrings,
    )


def test_chunk_id_differs_by_variant() -> None:
    full_id = _chunk_id("src/foo.py", "f", 1, strip_docstrings=False)
    stripped_id = _chunk_id("src/foo.py", "f", 1, strip_docstrings=True)
    assert full_id != stripped_id


def test_both_variants_coexist_in_same_store() -> None:
    """Same logical symbol indexed in both variants survives.

    `get_chunks(variant=...)` filters to one variant; both rows
    are retrievable by querying each variant in turn.
    """
    full = _make_chunk(
        content='def load_config():\n    """Load."""\n    pass\n',
        strip_docstrings=False,
    )
    stripped = _make_chunk(content="def load_config():\n    \n    pass\n", strip_docstrings=True)
    store = IndexStore()
    repo_id = store.register_repo("/test/repo")
    store.insert_chunks([full, stripped], repo_id=repo_id)
    store.insert_snapshot("HEAD", full.file_path, full.blob_sha, repo_id=repo_id)

    full_rows = store.get_chunks("HEAD", variant=IndexVariant.FULL, repo_id=repo_id)
    stripped_rows = store.get_chunks("HEAD", variant=IndexVariant.STRIPPED, repo_id=repo_id)

    full_by_id = {c.id: c for c in full_rows}
    stripped_by_id = {c.id: c for c in stripped_rows}
    assert full.id in full_by_id
    assert full.id not in stripped_by_id
    assert stripped.id in stripped_by_id
    assert stripped.id not in full_by_id
    assert full_by_id[full.id].strip_docstrings is False
    assert stripped_by_id[stripped.id].strip_docstrings is True
    assert "Load" in full_by_id[full.id].content
    assert "Load" not in stripped_by_id[stripped.id].content


def test_default_is_false() -> None:
    """Chunks without an explicit value default to strip_docstrings=False."""
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
