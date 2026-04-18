"""Behaviours of ``IndexStore`` that do not fit a scenario family.

What stays here:

- batch snapshots (single behaviour each)
- thread-safety / concurrency
- FTS internal invariants (IDF neutralisation, code-aware tokenisation)
- schema-check-with-existing-connection

Everything that was shared is now either a scenario in one of the
``case_store_*.py`` / ``test_store_*.py`` families or a fixture in
``conftest.py``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore, _check_schema_version
from rbtr.index.tokenise import tokenise_code


@pytest.fixture
def store() -> IndexStore:
    """In-memory store for tests that run against an empty store."""
    return IndexStore()


@pytest.fixture
def tokenised_chunks(all_store_chunks: list[Chunk]) -> list[Chunk]:
    """``all_store_chunks`` with content_tokens and name_tokens populated."""
    return [
        c.model_copy(
            update={
                "content_tokens": tokenise_code(c.content),
                "name_tokens": tokenise_code(c.name),
            }
        )
        for c in all_store_chunks
    ]


@pytest.fixture
def seeded_basic_store(store: IndexStore, tokenised_chunks: list[Chunk]) -> IndexStore:
    """Populated with every ``all_store_chunks`` chunk on commit 'head'."""
    store.insert_chunks(tokenised_chunks)
    for c in tokenised_chunks:
        store.insert_snapshot("head", c.file_path, c.blob_sha)
    return store


# ── Batch snapshots ─────────────────────────────────────────────────


def test_insert_snapshots_batch(store: IndexStore, math_func: Chunk, http_func: Chunk) -> None:
    store.insert_chunks([math_func, http_func])
    store.insert_snapshots(
        [
            ("sha1", math_func.file_path, math_func.blob_sha),
            ("sha1", http_func.file_path, http_func.blob_sha),
        ]
    )
    assert len(store.get_chunks("sha1")) == 2


def test_delete_snapshots_removes_ref_visibility(
    store: IndexStore, math_func: Chunk, http_func: Chunk
) -> None:
    """Deleting snapshots hides chunks at a ref without touching chunks."""
    store.insert_chunks([math_func, http_func])
    store.insert_snapshots(
        [
            ("head", math_func.file_path, math_func.blob_sha),
            ("head", http_func.file_path, http_func.blob_sha),
        ]
    )
    assert len(store.get_chunks("head")) == 2

    store.delete_snapshots("head")

    assert store.get_chunks("head") == []
    assert store.has_blob(math_func.blob_sha) is True
    assert store.has_blob(http_func.blob_sha) is True


# ── Thread safety ────────────────────────────────────────────────────


def test_concurrent_write_then_read(math_func: Chunk, http_func: Chunk, string_func: Chunk) -> None:
    """Writes from a background thread are visible after join."""
    store = IndexStore()
    store.insert_chunks([math_func, http_func, string_func])

    def writer() -> None:
        store.insert_snapshots(
            [
                ("head", math_func.file_path, math_func.blob_sha),
                ("head", http_func.file_path, http_func.blob_sha),
                ("head", string_func.file_path, string_func.blob_sha),
            ]
        )
        store.insert_edges(
            [Edge(source_id=math_func.id, target_id=http_func.id, kind=EdgeKind.IMPORTS)],
            "head",
        )

    t = threading.Thread(target=writer)
    t.start()
    t.join()

    assert len(store.get_chunks("head")) == 3
    assert len(store.get_edges("head")) == 1
    store.close()


def test_checkpoint_makes_writes_visible_during_concurrent_work(
    math_func: Chunk, http_func: Chunk, string_func: Chunk
) -> None:
    """After ``checkpoint()``, a second thread sees rows while the
    writer continues with slow updates."""
    store = IndexStore()
    checkpoint_done = threading.Event()
    updates_done = threading.Event()

    def writer() -> None:
        store.insert_chunks([math_func, http_func, string_func])
        store.insert_snapshots(
            [
                ("head", math_func.file_path, math_func.blob_sha),
                ("head", http_func.file_path, http_func.blob_sha),
                ("head", string_func.file_path, string_func.blob_sha),
            ]
        )
        store.checkpoint()
        checkpoint_done.set()
        store.update_embedding(math_func.id, [0.1, 0.2])
        store.update_embedding(http_func.id, [0.3, 0.4])
        store.update_embedding(string_func.id, [0.5, 0.6])
        updates_done.set()

    t = threading.Thread(target=writer)
    t.start()

    checkpoint_done.wait(timeout=5)
    assert len(store.get_chunks("head")) == 3

    t.join()
    store.close()


# ── FTS internal invariants ─────────────────────────────────────────


def test_idf_neutralised_after_fts_rebuild(
    seeded_basic_store: IndexStore,
) -> None:
    """After ``rebuild_fts_index()``, every term has df=1."""
    seeded_basic_store.rebuild_fts_index()
    rows = (
        seeded_basic_store._cur().execute("SELECT DISTINCT df FROM fts_main_chunks.dict").fetchall()
    )
    assert rows == [(1,)]


def test_fts_finds_camelcase_by_parts(store: IndexStore) -> None:
    """Splitting a CamelCase name lets the parts match."""
    chunk = Chunk(
        id="camel_1",
        blob_sha="b_camel",
        file_path="src/lib.py",
        kind=ChunkKind.CLASS,
        name="AgentDeps",
        content="class AgentDeps: pass",
        content_tokens=tokenise_code("class AgentDeps: pass"),
        name_tokens=tokenise_code("AgentDeps"),
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([chunk])
    store.insert_snapshot("head", chunk.file_path, chunk.blob_sha)
    store.rebuild_fts_index()

    results = store.search_fulltext("head", "agent deps", top_k=5)
    assert results
    assert results[0][0].id == "camel_1"

    results = store.search_fulltext("head", "AgentDeps", top_k=5)
    assert results
    assert results[0][0].id == "camel_1"


def test_fts_finds_snake_case_by_parts(store: IndexStore) -> None:
    """snake_case tokens are searchable individually."""
    chunk = Chunk(
        id="snake_1",
        blob_sha="b_snake",
        file_path="src/lib.py",
        kind=ChunkKind.FUNCTION,
        name="build_index",
        content="def build_index(repo): pass",
        content_tokens=tokenise_code("def build_index(repo): pass"),
        name_tokens=tokenise_code("build_index"),
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([chunk])
    store.insert_snapshot("head", chunk.file_path, chunk.blob_sha)

    results = store.search_fulltext("head", "build index", top_k=5)
    assert results
    assert results[0][0].id == "snake_1"


def test_fts_content_tokens_roundtrip(seeded_basic_store: IndexStore, math_func: Chunk) -> None:
    """``content_tokens`` set on insert is retrievable on read."""
    math = next(c for c in seeded_basic_store.get_chunks("head") if c.id == math_func.id)
    assert "calculate" in math.content_tokens
    assert "standard" in math.content_tokens
    assert "deviation" in math.content_tokens


# ── Schema check with an already-open connection ─────────────────────


def test_schema_check_skips_when_db_already_open(tmp_path: Path) -> None:
    """A second ``_check_schema_version`` against an open DB is a no-op."""
    db_path = tmp_path / "index.duckdb"
    store = IndexStore(db_path)
    _check_schema_version(db_path)
    store2 = IndexStore(db_path)
    store2.close()
    store.close()
