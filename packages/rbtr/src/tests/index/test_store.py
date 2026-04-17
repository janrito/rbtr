"""Tests for DuckDB index store.

Uses a data-first approach: insert realistic, semantically distinct
code chunks and verify that queries return correct results with
proper ranking.
"""

import threading
from pathlib import Path

import pytest

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore, _check_schema_version
from rbtr.index.tokenise import tokenise_code


@pytest.fixture
def store() -> IndexStore:
    """In-memory store for testing."""
    return IndexStore()


# ── Shared test data ────────────────────────────────────────────────
#
# Constants and the _seed_store helper were removed when the
# corresponding flat tests migrated to pytest-cases families.  What
# remains imports the few chunks still referenced by batch-snapshot,
# thread-safety, IDF, and code-aware FTS tests.

from tests.index.cases_common import HTTP_FUNC as _HTTP_FUNC
from tests.index.cases_common import MATH_CLASS as _MATH_CLASS
from tests.index.cases_common import MATH_FUNC as _MATH_FUNC
from tests.index.cases_common import STRING_FUNC as _STRING_FUNC
from tests.index.cases_common import VEC_HTTP as _VEC_HTTP
from tests.index.cases_common import VEC_MATH as _VEC_MATH
from tests.index.cases_common import VEC_STRING as _VEC_STRING

ALL_CHUNKS = [_MATH_FUNC, _HTTP_FUNC, _STRING_FUNC, _MATH_CLASS]


def _seed_store(store: IndexStore, *, embed: bool = False) -> None:
    """Insert all test chunks and snapshots into *store*."""
    for c in ALL_CHUNKS:
        c.content_tokens = tokenise_code(c.content)
        c.name_tokens = tokenise_code(c.name)
    store.insert_chunks(ALL_CHUNKS)
    for c in ALL_CHUNKS:
        store.insert_snapshot("head", c.file_path, c.blob_sha)
    if embed:
        store.update_embedding("math_1", _VEC_MATH)
        store.update_embedding("http_1", _VEC_HTTP)
        store.update_embedding("string_1", _VEC_STRING)
        store.update_embedding("math_class_1", _VEC_MATH)


# ── Batch snapshots ─────────────────────────────────────────────────


def test_insert_snapshots_batch(store: IndexStore) -> None:
    store.insert_chunks([_MATH_FUNC, _HTTP_FUNC])
    store.insert_snapshots(
        [
            ("sha1", "src/math_utils.py", "blob_math"),
            ("sha1", "src/api/client.py", "blob_http"),
        ]
    )
    assert len(store.get_chunks("sha1")) == 2


def test_delete_snapshots_removes_ref_visibility(store: IndexStore) -> None:
    """Deleting snapshots hides chunks at that ref without touching chunks table."""
    store.insert_chunks([_MATH_FUNC, _HTTP_FUNC])
    store.insert_snapshots(
        [
            ("head", "src/math_utils.py", "blob_math"),
            ("head", "src/api/client.py", "blob_http"),
        ]
    )
    assert len(store.get_chunks("head")) == 2

    store.delete_snapshots("head")

    # Ref view is empty.
    assert store.get_chunks("head") == []
    # Underlying chunks are still present (not pruned yet).
    assert store.has_blob("blob_math") is True
    assert store.has_blob("blob_http") is True


# ── Thread safety ────────────────────────────────────────────────────


def test_concurrent_write_then_read() -> None:
    """Writes from a background thread are visible after join."""

    store = IndexStore()
    store.insert_chunks([_MATH_FUNC, _HTTP_FUNC, _STRING_FUNC])

    def writer() -> None:
        store.insert_snapshots(
            [
                ("head", "src/math_utils.py", "blob_math"),
                ("head", "src/api/client.py", "blob_http"),
                ("head", "src/text/normalize.py", "blob_string"),
            ]
        )
        store.insert_edges(
            [Edge(source_id="math_1", target_id="http_1", kind=EdgeKind.IMPORTS)],
            "head",
        )

    t = threading.Thread(target=writer)
    t.start()
    t.join()

    chunks = store.get_chunks("head")
    assert len(chunks) == 3
    edges = store.get_edges("head")
    assert len(edges) == 1
    store.close()


def test_checkpoint_makes_writes_visible_during_concurrent_work() -> None:
    """After checkpoint(), the main thread sees data while the
    background thread continues with long-running updates.

    Reproduces the real-world scenario: indexing inserts chunks +
    snapshots, calls checkpoint(), then starts embedding (slow UPDATEs).
    `/index status` reads from the main thread during embedding.
    """

    store = IndexStore()
    checkpoint_done = threading.Event()
    updates_done = threading.Event()

    def writer() -> None:
        store.insert_chunks([_MATH_FUNC, _HTTP_FUNC, _STRING_FUNC])
        store.insert_snapshots(
            [
                ("head", "src/math_utils.py", "blob_math"),
                ("head", "src/api/client.py", "blob_http"),
                ("head", "src/text/normalize.py", "blob_string"),
            ]
        )
        store.checkpoint()
        checkpoint_done.set()
        # Simulate embedding: rapid UPDATE calls
        store.update_embedding("math_1", [0.1, 0.2])
        store.update_embedding("http_1", [0.3, 0.4])
        store.update_embedding("string_1", [0.5, 0.6])
        updates_done.set()

    t = threading.Thread(target=writer)
    t.start()

    # Wait for checkpoint, then read from main thread
    checkpoint_done.wait(timeout=5)
    chunks = store.get_chunks("head")
    assert len(chunks) == 3

    t.join()
    store.close()


# ── IDF neutralisation ──────────────────────────────────────────────


def test_idf_neutralised_after_fts_rebuild(store: IndexStore) -> None:
    """After rebuild_fts_index(), all terms have df=1."""
    _seed_store(store)
    store.rebuild_fts_index()

    df_values = store._cur().execute("SELECT DISTINCT df FROM fts_main_chunks.dict").fetchall()
    # Every term should have df=1 — only one distinct value.
    assert df_values == [(1,)]


# ── Code-aware FTS ───────────────────────────────────────────────────


def test_fts_finds_camelcase_by_parts(store: IndexStore) -> None:
    """Searching for split parts of a camelCase name returns the chunk."""
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

    # The tokeniser splits AgentDeps → agentdeps agent deps.
    # Searching for "agent deps" (two words) should find it.
    results = store.search_fulltext("head", "agent deps", top_k=5)
    assert len(results) >= 1
    assert results[0][0].id == "camel_1"

    # The compound form should also work.
    results = store.search_fulltext("head", "AgentDeps", top_k=5)
    assert len(results) >= 1
    assert results[0][0].id == "camel_1"


def test_fts_finds_snake_case_by_parts(store: IndexStore) -> None:
    """Searching for individual parts of a snake_case identifier works."""
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

    # Should find via individual parts.
    results = store.search_fulltext("head", "build index", top_k=5)
    assert len(results) >= 1
    assert results[0][0].id == "snake_1"


def test_fts_content_tokens_roundtrip(store: IndexStore) -> None:
    """Chunks with content_tokens populated survive insert → get."""
    _seed_store(store)
    chunks = store.get_chunks("head")
    math = next(c for c in chunks if c.id == "math_1")
    # content_tokens should contain tokenised content.
    assert "calculate" in math.content_tokens
    assert "standard" in math.content_tokens
    assert "deviation" in math.content_tokens


# ── Schema version check with existing connection ────────────────────


def test_schema_check_skips_when_db_already_open(tmp_path: Path) -> None:
    """_check_schema_version returns without error when the DB is already
    open with a read-write connection (different config than read_only=True)."""

    db_path = tmp_path / "index.duckdb"
    # First store opens a read-write connection.
    store = IndexStore(db_path)
    # Second schema check must not crash — the DB is already open.
    _check_schema_version(db_path)
    # Creating a second store on the same file also succeeds.
    store2 = IndexStore(db_path)
    store2.close()
    store.close()

