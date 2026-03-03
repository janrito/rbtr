"""Tests for DuckDB index store.

Uses a data-first approach: insert realistic, semantically distinct
code chunks and verify that queries return correct results with
proper ranking.
"""

from pathlib import Path

import duckdb
import pytest
from pytest_mock import MockerFixture

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import EMBEDDING_VERSION, SCHEMA_VERSION, IndexStore
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


# ── Basic roundtrip ─────────────────────────────────────────────────


def test_roundtrip_preserves_all_fields(store: IndexStore) -> None:
    """Every Chunk field survives insert → get_chunks."""
    _seed_store(store)
    chunks = store.get_chunks("head")
    by_id = {c.id: c for c in chunks}

    math = by_id["math_1"]
    assert math.name == "calculate_standard_deviation"
    assert math.kind == ChunkKind.FUNCTION
    assert math.file_path == "src/math_utils.py"
    assert math.blob_sha == "blob_math"
    assert math.line_start == 1
    assert math.line_end == 4
    assert "standard_deviation" in math.content
    assert math.scope == ""


def test_chunks_scoped_to_commit(store: IndexStore) -> None:
    _seed_store(store)
    assert len(store.get_chunks("head")) == 4
    assert len(store.get_chunks("other")) == 0


def test_has_blob(store: IndexStore) -> None:
    _seed_store(store)
    assert store.has_blob("blob_math")
    assert not store.has_blob("nonexistent")


def test_filter_by_file_path(store: IndexStore) -> None:
    _seed_store(store)
    result = store.get_chunks("head", file_path="src/api/client.py")
    assert len(result) == 1
    assert result[0].id == "http_1"


def test_filter_by_kind(store: IndexStore) -> None:
    _seed_store(store)
    classes = store.get_chunks("head", kind=ChunkKind.CLASS)
    assert len(classes) == 1
    assert classes[0].id == "math_class_1"


def test_filter_by_name(store: IndexStore) -> None:
    _seed_store(store)
    result = store.get_chunks("head", name="normalize_whitespace")
    assert len(result) == 1
    assert result[0].id == "string_1"


def test_upsert_updates_content(store: IndexStore) -> None:
    """Inserting a chunk with the same ID updates its content."""
    _seed_store(store)
    updated = _MATH_FUNC.model_copy(update={"content": "def calculate(): return 42"})
    store.insert_chunks([updated])
    result = store.get_chunks("head", name="calculate_standard_deviation")
    assert "return 42" in result[0].content


def test_blob_reuse_across_commits(store: IndexStore) -> None:
    """Same blob_sha in two commits → stored once, visible in both."""
    store.insert_chunks([_MATH_FUNC])
    store.insert_snapshot("commit_a", "src/math_utils.py", "blob_math")
    store.insert_snapshot("commit_b", "src/math_utils.py", "blob_math")
    assert len(store.get_chunks("commit_a")) == 1
    assert len(store.get_chunks("commit_b")) == 1


# ── Edges ────────────────────────────────────────────────────────────


def test_edge_roundtrip(store: IndexStore) -> None:
    edge = Edge(source_id="math_1", target_id="math_class_1", kind=EdgeKind.CALLS)
    store.insert_edges([edge], "head")
    result = store.get_edges("head")
    assert len(result) == 1
    assert result[0] == edge


def test_edges_scoped_to_commit(store: IndexStore) -> None:
    store.insert_edges([Edge(source_id="a", target_id="b", kind=EdgeKind.IMPORTS)], "head")
    assert len(store.get_edges("head")) == 1
    assert len(store.get_edges("other")) == 0


def test_edge_filter_by_source(store: IndexStore) -> None:
    store.insert_edges(
        [
            Edge(source_id="a", target_id="b", kind=EdgeKind.CALLS),
            Edge(source_id="c", target_id="d", kind=EdgeKind.IMPORTS),
        ],
        "head",
    )
    result = store.get_edges("head", source_id="a")
    assert len(result) == 1
    assert result[0].source_id == "a"


def test_edge_filter_by_kind(store: IndexStore) -> None:
    store.insert_edges(
        [
            Edge(source_id="a", target_id="b", kind=EdgeKind.CALLS),
            Edge(source_id="c", target_id="d", kind=EdgeKind.IMPORTS),
        ],
        "head",
    )
    result = store.get_edges("head", kind=EdgeKind.IMPORTS)
    assert len(result) == 1
    assert result[0].kind == EdgeKind.IMPORTS


# ── Diff ─────────────────────────────────────────────────────────────


def test_diff_detects_added_file(store: IndexStore) -> None:
    store.insert_chunks([_MATH_FUNC, _HTTP_FUNC])
    store.insert_snapshot("base", "src/math_utils.py", "blob_math")
    store.insert_snapshot("head", "src/math_utils.py", "blob_math")
    store.insert_snapshot("head", "src/api/client.py", "blob_http")

    added, removed, modified = store.diff_chunks("base", "head")
    assert [c.id for c in added] == ["http_1"]
    assert removed == []
    assert modified == []


def test_diff_detects_removed_file(store: IndexStore) -> None:
    store.insert_chunks([_MATH_FUNC, _HTTP_FUNC])
    store.insert_snapshot("base", "src/math_utils.py", "blob_math")
    store.insert_snapshot("base", "src/api/client.py", "blob_http")
    store.insert_snapshot("head", "src/math_utils.py", "blob_math")

    added, removed, modified = store.diff_chunks("base", "head")
    assert added == []
    assert [c.id for c in removed] == ["http_1"]
    assert modified == []


def test_diff_detects_modified_file(store: IndexStore) -> None:
    old = _MATH_FUNC
    new = _MATH_FUNC.model_copy(update={"id": "math_1_v2", "blob_sha": "blob_math_v2"})
    store.insert_chunks([old, new])
    store.insert_snapshot("base", "src/math_utils.py", "blob_math")
    store.insert_snapshot("head", "src/math_utils.py", "blob_math_v2")

    added, removed, modified = store.diff_chunks("base", "head")
    assert added == []
    assert removed == []
    assert len(modified) == 1
    assert modified[0].id == "math_1_v2"


# ── Embedding roundtrip ─────────────────────────────────────────────


def test_embedding_absent_is_falsy(store: IndexStore) -> None:
    _seed_store(store, embed=False)
    chunks = store.get_chunks("head")
    assert all(not c.embedding for c in chunks)


def test_embedding_present_is_truthy(store: IndexStore) -> None:
    _seed_store(store, embed=True)
    chunks = store.get_chunks("head")
    assert all(c.embedding for c in chunks)


def test_embedding_via_update(store: IndexStore) -> None:
    _seed_store(store, embed=False)
    store.update_embedding("math_1", [0.1, 0.2, 0.3])

    chunks = store.get_chunks("head")
    by_id = {c.id: c for c in chunks}
    assert by_id["math_1"].embedding  # truthy
    assert not by_id["http_1"].embedding  # still absent


def test_batch_update_embeddings(store: IndexStore) -> None:
    """update_embeddings sets vectors for multiple chunks in one call."""
    _seed_store(store, embed=False)
    store.update_embeddings(
        ["math_1", "http_1"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    chunks = store.get_chunks("head")
    by_id = {c.id: c for c in chunks}
    assert by_id["math_1"].embedding  # truthy
    assert by_id["http_1"].embedding  # truthy
    assert not by_id["string_1"].embedding  # untouched


def test_batch_update_embeddings_empty(store: IndexStore) -> None:
    """update_embeddings with empty lists is a no-op."""
    _seed_store(store, embed=False)
    store.update_embeddings([], [])  # should not raise


def test_clear_embeddings(store: IndexStore) -> None:
    """clear_embeddings nulls all vectors and returns the count."""
    _seed_store(store, embed=True)

    cleared = store.clear_embeddings()
    assert cleared == 4  # math, http, string, math_class

    chunks = store.get_chunks("head")
    assert all(not c.embedding for c in chunks)


def test_clear_embeddings_when_none(store: IndexStore) -> None:
    """clear_embeddings on unembedded chunks returns 0."""
    _seed_store(store, embed=False)

    cleared = store.clear_embeddings()
    assert cleared == 0


# ── Similarity search — ranking ──────────────────────────────────────


def test_similar_ranks_closest_first(store: IndexStore) -> None:
    """Query near the math axis ranks math chunks above HTTP/string."""
    _seed_store(store, embed=True)
    query = [0.9, 0.1, 0.0]  # close to math, far from string

    results = store.search_similar("head", query, top_k=4)
    ids = [c.id for c, _ in results]

    # Math chunks should come first (both share _VEC_MATH).
    assert ids[0] in ("math_1", "math_class_1")
    assert ids[1] in ("math_1", "math_class_1")
    # String should be last (orthogonal to query).
    assert ids[-1] == "string_1"


def test_similar_exact_match_scores_near_one(store: IndexStore) -> None:
    """Querying with the exact same vector gives score ≈ 1.0."""
    _seed_store(store, embed=True)
    results = store.search_similar("head", _VEC_HTTP, top_k=1)
    chunk, score = results[0]
    assert chunk.id == "http_1"
    assert score > 0.99


def test_similar_orthogonal_scores_near_zero(store: IndexStore) -> None:
    """Orthogonal vectors give score ≈ 0.0."""
    _seed_store(store, embed=True)
    results = store.search_similar("head", _VEC_STRING, top_k=4)
    # Math chunks have [1,0,0], query is [0,0,1] — cosine = 0.
    math_scores = [s for c, s in results if c.id == "math_1"]
    assert math_scores[0] < 0.01


def test_similar_respects_commit_scope(store: IndexStore) -> None:
    _seed_store(store, embed=True)
    results = store.search_similar("nonexistent_commit", _VEC_MATH, top_k=4)
    assert results == []


def test_similar_skips_unembedded_chunks(store: IndexStore) -> None:
    """Chunks without embeddings are excluded from similarity results."""
    _seed_store(store, embed=False)
    store.update_embedding("math_1", _VEC_MATH)
    # Only math_1 has an embedding.
    results = store.search_similar("head", _VEC_MATH, top_k=10)
    assert len(results) == 1
    assert results[0][0].id == "math_1"


def test_similar_result_has_embedding_flag(store: IndexStore) -> None:
    _seed_store(store, embed=True)
    results = store.search_similar("head", _VEC_MATH, top_k=1)
    assert results[0][0].embedding  # truthy


# ── Search by name ───────────────────────────────────────────────────


def test_search_by_name_finds_substring(store: IndexStore) -> None:
    _seed_store(store)
    result = store.search_by_name("head", "standard_deviation")
    assert len(result) == 1
    assert result[0].id == "math_1"


def test_search_by_name_case_insensitive(store: IndexStore) -> None:
    _seed_store(store)
    result = store.search_by_name("head", "NORMALIZE")
    assert len(result) == 1
    assert result[0].id == "string_1"


def test_search_by_name_no_match(store: IndexStore) -> None:
    _seed_store(store)
    assert store.search_by_name("head", "zzz_nonexistent") == []


def test_search_by_name_has_embedding_flag(store: IndexStore) -> None:
    _seed_store(store, embed=True)
    result = store.search_by_name("head", "fetch_json")
    assert len(result) == 1
    assert result[0].embedding


# ── Full-text search — ranking ───────────────────────────────────────


def test_fulltext_ranks_by_relevance(store: IndexStore) -> None:
    """Searching for 'variance' ranks the math function highest."""
    _seed_store(store)
    store.rebuild_fts_index()

    results = store.search_fulltext("head", "variance", top_k=4)
    assert len(results) >= 1
    assert results[0][0].id == "math_1"


def test_fulltext_finds_http_content(store: IndexStore) -> None:
    """Searching for 'endpoint' finds the HTTP function."""
    _seed_store(store)
    store.rebuild_fts_index()

    results = store.search_fulltext("head", "endpoint", top_k=4)
    assert len(results) >= 1
    ids = [c.id for c, _ in results]
    assert "http_1" in ids


def test_fulltext_finds_regex_content(store: IndexStore) -> None:
    """Searching for 'whitespace' finds the string function."""
    _seed_store(store)
    store.rebuild_fts_index()

    results = store.search_fulltext("head", "whitespace", top_k=4)
    assert len(results) >= 1
    ids = [c.id for c, _ in results]
    assert "string_1" in ids


def test_fulltext_scoped_to_commit(store: IndexStore) -> None:
    _seed_store(store)
    store.rebuild_fts_index()

    assert len(store.search_fulltext("head", "variance", top_k=4)) >= 1
    assert store.search_fulltext("other_commit", "variance", top_k=4) == []


def test_fulltext_no_match(store: IndexStore) -> None:
    _seed_store(store)
    store.rebuild_fts_index()
    assert store.search_fulltext("head", "zzz_gibberish_xyz", top_k=4) == []


def test_fulltext_result_has_embedding_flag(store: IndexStore) -> None:
    _seed_store(store, embed=True)
    store.rebuild_fts_index()

    results = store.search_fulltext("head", "variance", top_k=1)
    assert len(results) >= 1
    assert results[0][0].embedding


def test_rebuild_fts_idempotent(store: IndexStore) -> None:
    store.rebuild_fts_index()
    store.rebuild_fts_index()  # no error


def test_fulltext_auto_rebuilds_fts_on_first_call(store: IndexStore) -> None:
    """search_fulltext works without an explicit rebuild_fts_index call.

    Reproduces the real-world bug where the FTS index is lost after
    reopening a DuckDB file (FTS is in-memory only).  The lazy
    ``_ensure_fts()`` must rebuild it transparently.
    """
    _seed_store(store)
    # Do NOT call rebuild_fts_index — that's the point.
    results = store.search_fulltext("head", "variance", top_k=4)
    assert len(results) >= 1
    assert results[0][0].name == "calculate_standard_deviation"


def test_fulltext_auto_rebuilds_after_new_chunks(store: IndexStore) -> None:
    """Inserting new chunks invalidates FTS; next search rebuilds it."""
    # Force an initial FTS build so _fts_dirty is False.
    store.rebuild_fts_index()
    assert not store._fts_dirty  # testing internal state

    # Insert a new chunk — should mark FTS dirty.
    new_chunk = Chunk(
        id="new_1",
        blob_sha="blob_new",
        file_path="src/new.py",
        kind=ChunkKind.FUNCTION,
        name="brand_new_function",
        scope="",
        content="def brand_new_function(): pass",
        content_tokens=tokenise_code("def brand_new_function(): pass"),
        name_tokens=tokenise_code("brand_new_function"),
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([new_chunk])
    store.insert_snapshot("head", new_chunk.file_path, new_chunk.blob_sha)
    assert store._fts_dirty  # testing internal state

    # Search should find the new chunk without an explicit rebuild.
    results = store.search_fulltext("head", "brand_new_function", top_k=4)
    assert len(results) >= 1
    assert results[0][0].name == "brand_new_function"
    assert not store._fts_dirty  # auto-rebuilt


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
    import threading

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
    ``/index status`` reads from the main thread during embedding.
    """
    import threading

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


# ── Orphan pruning ───────────────────────────────────────────────────


def test_count_orphan_chunks_zero_when_all_referenced(store: IndexStore) -> None:
    """No orphans when every chunk is referenced by a snapshot."""
    store.insert_chunks([_MATH_FUNC, _HTTP_FUNC])
    store.insert_snapshots(
        [
            ("sha1", "src/math_utils.py", "blob_math"),
            ("sha1", "src/api/client.py", "blob_http"),
        ]
    )
    assert store.count_orphan_chunks() == 0


def test_count_orphan_chunks_detects_unreferenced(store: IndexStore) -> None:
    """Chunks not referenced by any snapshot are counted as orphans."""
    store.insert_chunks([_MATH_FUNC, _HTTP_FUNC, _STRING_FUNC])
    # Only reference math — http and string are orphans.
    store.insert_snapshots([("sha1", "src/math_utils.py", "blob_math")])
    assert store.count_orphan_chunks() == 2


def test_prune_orphans_deletes_unreferenced_chunks(store: IndexStore) -> None:
    """prune_orphans removes chunks not referenced by any snapshot."""
    store.insert_chunks([_MATH_FUNC, _HTTP_FUNC, _STRING_FUNC])
    store.insert_snapshots([("sha1", "src/math_utils.py", "blob_math")])

    chunks_del, edges_del = store.prune_orphans()
    assert chunks_del == 2
    assert edges_del == 0

    # Only the referenced chunk remains.
    remaining = store.get_chunks("sha1")
    assert len(remaining) == 1
    assert remaining[0].id == "math_1"

    # Orphan count is now zero.
    assert store.count_orphan_chunks() == 0


def test_prune_orphans_deletes_unreferenced_edges(store: IndexStore) -> None:
    """prune_orphans removes edges whose commit_sha has no snapshots."""
    store.insert_chunks([_MATH_FUNC, _HTTP_FUNC])
    store.insert_snapshots(
        [
            ("sha1", "src/math_utils.py", "blob_math"),
            ("sha1", "src/api/client.py", "blob_http"),
        ]
    )
    # Edges for "sha1" (referenced) and "stale_sha" (no snapshots).
    store.insert_edges(
        [Edge(source_id="math_1", target_id="http_1", kind=EdgeKind.IMPORTS)],
        "sha1",
    )
    store.insert_edges(
        [Edge(source_id="math_1", target_id="http_1", kind=EdgeKind.IMPORTS)],
        "stale_sha",
    )

    chunks_del, edges_del = store.prune_orphans()
    assert chunks_del == 0
    assert edges_del == 1

    # Only edges for sha1 remain.
    assert len(store.get_edges("sha1")) == 1
    assert len(store.get_edges("stale_sha")) == 0


def test_prune_orphans_noop_when_clean(store: IndexStore) -> None:
    """prune_orphans returns zeros when no orphans exist."""
    store.insert_chunks([_MATH_FUNC])
    store.insert_snapshots([("sha1", "src/math_utils.py", "blob_math")])
    store.insert_edges(
        [Edge(source_id="math_1", target_id="math_1", kind=EdgeKind.IMPORTS)],
        "sha1",
    )
    chunks_del, edges_del = store.prune_orphans()
    assert chunks_del == 0
    assert edges_del == 0


def test_prune_marks_fts_dirty(store: IndexStore) -> None:
    """After pruning chunks, FTS index is rebuilt on next search."""
    for c in [_MATH_FUNC, _HTTP_FUNC]:
        c.content_tokens = tokenise_code(c.content)
        c.name_tokens = tokenise_code(c.name)
    store.insert_chunks([_MATH_FUNC, _HTTP_FUNC])
    store.insert_snapshots([("sha1", "src/math_utils.py", "blob_math")])
    store.rebuild_fts_index()
    assert not store._fts_dirty

    store.prune_orphans()
    assert store._fts_dirty

    # Fulltext search still works (triggers lazy rebuild).
    results = store.search_fulltext("sha1", "deviation")
    assert len(results) == 1


def test_prune_after_file_change(store: IndexStore) -> None:
    """Simulates a file changing between commits — old chunks become orphans."""
    # v1: math blob
    old_math = Chunk(
        id="math_v1",
        blob_sha="blob_math_v1",
        file_path="src/math_utils.py",
        kind=ChunkKind.FUNCTION,
        name="calculate",
        content="def calculate(): pass",
        line_start=1,
        line_end=1,
    )
    # v2: same path, new blob
    new_math = Chunk(
        id="math_v2",
        blob_sha="blob_math_v2",
        file_path="src/math_utils.py",
        kind=ChunkKind.FUNCTION,
        name="calculate_v2",
        content="def calculate_v2(): pass",
        line_start=1,
        line_end=1,
    )

    store.insert_chunks([old_math])
    store.insert_snapshots([("base", "src/math_utils.py", "blob_math_v1")])

    # Now head has the new version — update snapshot.
    store.insert_chunks([new_math])
    store.insert_snapshots([("head", "src/math_utils.py", "blob_math_v2")])

    # Remove old snapshot (simulating a re-index that only keeps head).
    store._cur().execute("DELETE FROM file_snapshots WHERE commit_sha = 'base'")

    # Old chunk is now orphaned.
    assert store.count_orphan_chunks() == 1
    chunks_del, _ = store.prune_orphans()
    assert chunks_del == 1

    # New chunk is still accessible.
    remaining = store.get_chunks("head")
    assert len(remaining) == 1
    assert remaining[0].id == "math_v2"


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
    mocker.patch("rbtr.index.store.config.index.embedding_model", "other/model.gguf")

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
    from rbtr.index.store import _check_schema_version

    db_path = tmp_path / "index.duckdb"
    # First store opens a read-write connection.
    store = IndexStore(db_path)
    # Second schema check must not crash — the DB is already open.
    _check_schema_version(db_path)
    # Creating a second store on the same file also succeeds.
    store2 = IndexStore(db_path)
    store2.close()
    store.close()
