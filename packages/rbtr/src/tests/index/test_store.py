"""Tests for DuckDB index store.

Uses a data-first approach: insert realistic, semantically distinct
code chunks and verify that queries return correct results with
proper ranking.
"""

import threading
from pathlib import Path

import duckdb
import pytest
from pytest_mock import MockerFixture

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import (
    EMBEDDING_VERSION,
    SCHEMA_VERSION,
    IndexStore,
    _check_schema_version,
)
from rbtr.index.tokenise import tokenise_code


@pytest.fixture
def store() -> IndexStore:
    """In-memory store for testing."""
    return IndexStore()


# ── Test data ────────────────────────────────────────────────────────
#
# Three clearly distinct functions: math, HTTP, and string processing.
# Each has unique vocabulary so FTS can distinguish them, and each
# gets a unit-axis embedding vector for cosine similarity tests.

_MATH_FUNC = Chunk(
    id="math_1",
    blob_sha="blob_math",
    file_path="src/math_utils.py",
    kind=ChunkKind.FUNCTION,
    name="calculate_standard_deviation",
    content="""\
def calculate_standard_deviation(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5
""",
    line_start=1,
    line_end=4,
)

_HTTP_FUNC = Chunk(
    id="http_1",
    blob_sha="blob_http",
    file_path="src/api/client.py",
    kind=ChunkKind.FUNCTION,
    name="fetch_json_from_endpoint",
    content="""\
async def fetch_json_from_endpoint(url: str, headers: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
""",
    line_start=10,
    line_end=15,
)

_STRING_FUNC = Chunk(
    id="string_1",
    blob_sha="blob_string",
    file_path="src/text/normalize.py",
    kind=ChunkKind.FUNCTION,
    name="normalize_whitespace",
    content="""\
def normalize_whitespace(text: str) -> str:
    import re
    collapsed = re.sub(r'\\s+', ' ', text)
    return collapsed.strip()
""",
    line_start=1,
    line_end=4,
)

_MATH_CLASS = Chunk(
    id="math_class_1",
    blob_sha="blob_math",  # same file as _MATH_FUNC → same blob
    file_path="src/math_utils.py",
    kind=ChunkKind.CLASS,
    name="StatisticsCalculator",
    content="""\
class StatisticsCalculator:
    def __init__(self, data: list[float]):
        self.data = data
    def mean(self) -> float:
        return sum(self.data) / len(self.data)
""",
    line_start=10,
    line_end=15,
)

# Embedding vectors — orthogonal unit axes so cosine similarity
# gives clean 0.0 / 1.0 values for unambiguous ranking.
_VEC_MATH = [1.0, 0.0, 0.0]
_VEC_HTTP = [0.0, 1.0, 0.0]
_VEC_STRING = [0.0, 0.0, 1.0]

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


# ── Schema versioning ───────────────────────────────────────────────


def test_schema_version_mismatch_deletes_db(tmp_path: Path) -> None:
    """Opening a DB with a stale schema version deletes the file."""
    db_path = tmp_path / "test.duckdb"

    # Create a store at the current version.
    s1 = IndexStore(db_path)
    s1.close()
    assert db_path.exists()

    # Simulate a version bump by writing a stale version to meta.
    con = duckdb.connect(str(db_path))
    con.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    con.close()

    # Re-opening should detect the mismatch, delete, and recreate.
    s2 = IndexStore(db_path)
    # The DB was nuked and recreated — meta should have the current version.
    rows = s2._cur().execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchall()
    assert rows[0][0] == str(SCHEMA_VERSION)
    s2.close()


def test_schema_version_missing_meta_table_deletes_db(tmp_path: Path) -> None:
    """A DB without a meta table (old schema) is deleted on open."""
    db_path = tmp_path / "old.duckdb"

    # Create a bare DuckDB file without our schema.
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE dummy (x INT)")
    con.close()
    assert db_path.exists()

    # Opening as IndexStore should nuke it.
    s = IndexStore(db_path)
    # Verify it's a fresh DB with the meta table.
    rows = s._cur().execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchall()
    assert rows[0][0] == str(SCHEMA_VERSION)
    s.close()


def test_schema_version_correct_keeps_data(tmp_path: Path) -> None:
    """When version matches, existing data is preserved."""
    db_path = tmp_path / "ok.duckdb"

    s1 = IndexStore(db_path)
    s1.insert_chunks([_MATH_FUNC])
    s1.insert_snapshot("head", _MATH_FUNC.file_path, _MATH_FUNC.blob_sha)
    s1.close()

    s2 = IndexStore(db_path)
    chunks = s2.get_chunks("head")
    assert len(chunks) == 1
    assert chunks[0].id == "math_1"
    s2.close()


# ── Embedding version ────────────────────────────────────────────────


def test_embedding_version_mismatch_clears_embeddings(tmp_path: Path) -> None:
    """Stale embedding version clears embeddings without nuking the DB."""
    db_path = tmp_path / "emb.duckdb"
    s1 = IndexStore(db_path)
    s1.insert_chunks([_MATH_FUNC])
    s1.insert_snapshot("head", _MATH_FUNC.file_path, _MATH_FUNC.blob_sha)
    s1.update_embeddings([_MATH_FUNC.id], [[0.1, 0.2, 0.3]])
    s1.close()

    # Simulate a stale embedding version.
    con = duckdb.connect(str(db_path))
    con.execute("UPDATE meta SET value = '0' WHERE key = 'embedding_version'")
    con.close()

    s2 = IndexStore(db_path)
    # Chunks should still exist.
    chunks = s2.get_chunks("head")
    assert len(chunks) == 1
    # But embeddings should be cleared.
    assert not chunks[0].embedding
    # Version should be stamped.
    rows = s2._cur().execute("SELECT value FROM meta WHERE key = 'embedding_version'").fetchall()
    assert rows[0][0] == str(EMBEDDING_VERSION)
    s2.close()


def test_embedding_version_correct_keeps_embeddings(tmp_path: Path) -> None:
    """When embedding version matches, embeddings are preserved."""
    db_path = tmp_path / "emb_ok.duckdb"
    s1 = IndexStore(db_path)
    s1.insert_chunks([_MATH_FUNC])
    s1.insert_snapshot("head", _MATH_FUNC.file_path, _MATH_FUNC.blob_sha)
    s1.update_embeddings([_MATH_FUNC.id], [[0.1, 0.2, 0.3]])
    s1.close()

    s2 = IndexStore(db_path)
    chunks = s2.get_chunks("head")
    assert len(chunks) == 1
    assert chunks[0].embedding  # truthy — embedding preserved
    s2.close()


def test_embedding_model_change_clears_embeddings(tmp_path: Path, mocker: MockerFixture) -> None:
    """Changing the configured embedding model clears embeddings."""
    db_path = tmp_path / "emb_model.duckdb"
    s1 = IndexStore(db_path)
    s1.insert_chunks([_MATH_FUNC])
    s1.insert_snapshot("head", _MATH_FUNC.file_path, _MATH_FUNC.blob_sha)
    s1.update_embeddings([_MATH_FUNC.id], [[0.1, 0.2, 0.3]])
    s1.close()

    # Simulate a config change to a different model.
    mocker.patch("rbtr.index.store.config.embedding_model", "other/model.gguf")

    s2 = IndexStore(db_path)
    chunks = s2.get_chunks("head")
    assert len(chunks) == 1
    assert not chunks[0].embedding  # cleared
    # Model should be stamped.
    rows = s2._cur().execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchall()
    assert rows[0][0] == "other/model.gguf"
    s2.close()


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


# ── Multi-repo isolation ───────────────────────────────────────────────


def test_register_repo_returns_id(store: IndexStore) -> None:
    """register_repo returns an integer ID."""
    repo_id = store.register_repo("/path/to/repo-a")
    assert isinstance(repo_id, int)
    assert repo_id >= 1


def test_register_repo_idempotent(store: IndexStore) -> None:
    """Registering the same path twice returns the same ID."""
    id1 = store.register_repo("/path/to/repo")
    id2 = store.register_repo("/path/to/repo")
    assert id1 == id2


def test_register_repo_distinct_ids(store: IndexStore) -> None:
    """Different paths get different IDs."""
    id_a = store.register_repo("/repo-a")
    id_b = store.register_repo("/repo-b")
    assert id_a != id_b


def test_list_repos(store: IndexStore) -> None:
    """list_repos returns all registered repos."""
    store.register_repo("/repo-a")
    store.register_repo("/repo-b")
    repos = store.list_repos()
    assert len(repos) == 2
    assert {path for _, path in repos} == {"/repo-a", "/repo-b"}


def test_chunks_isolated_by_repo(store: IndexStore) -> None:
    """Chunks from different repos don't leak across repo_id boundaries."""
    id_a = store.register_repo("/repo-a")
    id_b = store.register_repo("/repo-b")

    chunk_a = Chunk(
        id="a1",
        blob_sha="ba",
        file_path="src/a.py",
        kind=ChunkKind.FUNCTION,
        name="func_a",
        content="def func_a(): pass",
        line_start=1,
        line_end=1,
    )
    chunk_b = Chunk(
        id="b1",
        blob_sha="bb",
        file_path="src/b.py",
        kind=ChunkKind.FUNCTION,
        name="func_b",
        content="def func_b(): pass",
        line_start=1,
        line_end=1,
    )

    store.insert_chunks([chunk_a], repo_id=id_a)
    store.insert_chunks([chunk_b], repo_id=id_b)
    store.insert_snapshot("HEAD", "src/a.py", "ba", repo_id=id_a)
    store.insert_snapshot("HEAD", "src/b.py", "bb", repo_id=id_b)

    results_a = store.get_chunks("HEAD", repo_id=id_a)
    results_b = store.get_chunks("HEAD", repo_id=id_b)

    assert len(results_a) == 1
    assert results_a[0].name == "func_a"
    assert len(results_b) == 1
    assert results_b[0].name == "func_b"


def test_edges_isolated_by_repo(store: IndexStore) -> None:
    """Edges from different repos don't leak."""
    id_a = store.register_repo("/repo-a")
    id_b = store.register_repo("/repo-b")

    edge_a = Edge(source_id="s_a", target_id="t_a", kind=EdgeKind.IMPORTS)
    edge_b = Edge(source_id="s_b", target_id="t_b", kind=EdgeKind.TESTS)

    store.insert_edges([edge_a], "HEAD", repo_id=id_a)
    store.insert_edges([edge_b], "HEAD", repo_id=id_b)

    edges_a = store.get_edges("HEAD", repo_id=id_a)
    edges_b = store.get_edges("HEAD", repo_id=id_b)

    assert len(edges_a) == 1
    assert edges_a[0].source_id == "s_a"
    assert len(edges_b) == 1
    assert edges_b[0].source_id == "s_b"


def test_search_by_name_isolated_by_repo(store: IndexStore) -> None:
    """Name search only returns chunks from the queried repo."""
    id_a = store.register_repo("/repo-a")
    id_b = store.register_repo("/repo-b")

    chunk_a = Chunk(
        id="a1",
        blob_sha="ba",
        file_path="src/a.py",
        kind=ChunkKind.CLASS,
        name="Config",
        content="class Config: pass",
        line_start=1,
        line_end=1,
    )
    chunk_b = Chunk(
        id="b1",
        blob_sha="bb",
        file_path="src/b.py",
        kind=ChunkKind.CLASS,
        name="Config",
        content="class Config: pass",
        line_start=1,
        line_end=1,
    )

    store.insert_chunks([chunk_a], repo_id=id_a)
    store.insert_chunks([chunk_b], repo_id=id_b)
    store.insert_snapshot("HEAD", "src/a.py", "ba", repo_id=id_a)
    store.insert_snapshot("HEAD", "src/b.py", "bb", repo_id=id_b)

    results = store.search_by_name("HEAD", "Config", repo_id=id_a)
    assert len(results) == 1
    assert results[0].id == "a1"


def test_has_blob_isolated_by_repo(store: IndexStore) -> None:
    """has_blob is repo-scoped."""
    id_a = store.register_repo("/repo-a")
    id_b = store.register_repo("/repo-b")

    chunk = Chunk(
        id="c1",
        blob_sha="blob_x",
        file_path="f.py",
        kind=ChunkKind.FUNCTION,
        name="f",
        content="pass",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([chunk], repo_id=id_a)

    assert store.has_blob("blob_x", repo_id=id_a)
    assert not store.has_blob("blob_x", repo_id=id_b)


def test_count_chunks_by_repo(store: IndexStore) -> None:
    """count_chunks returns count without loading data."""
    id_a = store.register_repo("/repo-a")
    id_b = store.register_repo("/repo-b")

    chunk = Chunk(
        id="c1",
        blob_sha="b1",
        file_path="f.py",
        kind=ChunkKind.FUNCTION,
        name="f",
        content="pass",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([chunk], repo_id=id_a)
    store.insert_snapshot("HEAD", "f.py", "b1", repo_id=id_a)

    assert store.count_chunks("HEAD", repo_id=id_a) == 1
    assert store.count_chunks("HEAD", repo_id=id_b) == 0

