"""DuckDB storage — schema, read/write, and search for the code index.

The store manages three tables:

- ``file_snapshots`` maps a commit SHA to its file tree (path → blob SHA).
- ``chunks`` holds indexed content, keyed by blob SHA so identical files
  across commits are stored once.
- ``edges`` records relationships between chunks, scoped per commit.

All commit-scoped queries join through ``file_snapshots`` to resolve
which chunks belong to a given snapshot.
"""

from __future__ import annotations

import contextlib
import importlib.resources
import json
import threading
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]  # no stubs available

from rbtr.index.arrow import chunks_to_table, edges_to_table, snapshots_to_table
from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind, ImportMeta

# ── SQL loader ───────────────────────────────────────────────────────

_sql_pkg = importlib.resources.files("rbtr.index") / "sql"


def _load_sql(name: str) -> str:
    """Read a .sql file from the sql/ package directory."""
    return (_sql_pkg / name).read_text(encoding="utf-8")


# Pre-load all SQL at import time so file I/O is not on the hot path.
_SCHEMA_SQL = _load_sql("schema.sql")
_GET_CHUNKS_SQL = _load_sql("get_chunks.sql")
_GET_EDGES_SQL = _load_sql("get_edges.sql")
_DIFF_ADDED_SQL = _load_sql("diff_added.sql")
_DIFF_MODIFIED_SQL = _load_sql("diff_modified.sql")
_SEARCH_BY_NAME_SQL = _load_sql("search_by_name.sql")
_SEARCH_SIMILAR_SQL = _load_sql("search_similar.sql")
_SEARCH_FULLTEXT_SQL = _load_sql("search_fulltext.sql")
_UPSERT_SNAPSHOTS_SQL = _load_sql("upsert_snapshots.sql")
_UPSERT_CHUNKS_SQL = _load_sql("upsert_chunks.sql")
_INSERT_EDGES_SQL = _load_sql("insert_edges.sql")
_PRUNE_CHUNKS_SQL = _load_sql("prune_chunks.sql")
_PRUNE_EDGES_SQL = _load_sql("prune_edges.sql")
_COUNT_ORPHAN_CHUNKS_SQL = _load_sql("count_orphan_chunks.sql")
_INSERT_SNAPSHOT_SQL = _load_sql("insert_snapshot.sql")
_DELETE_EDGES_SQL = _load_sql("delete_edges.sql")
_UPDATE_EMBEDDING_SQL = _load_sql("update_embedding.sql")
_UPDATE_EMBEDDINGS_SQL = _load_sql("update_embeddings.sql")
_HAS_BLOB_SQL = _load_sql("has_blob.sql")
_CLEAR_EMBEDDINGS_SQL = _load_sql("clear_embeddings.sql")

# diff_removed is the same query as diff_added with swapped params.
_DIFF_REMOVED_SQL = _DIFF_ADDED_SQL


# ── Row mapping ──────────────────────────────────────────────────────

_EMBEDDING_SENTINEL = [0.0]
"""Lightweight marker for 'has embedding in DB but not loaded'.

``Chunk.embedding`` is ``list[float]`` — empty means absent.
Loading full 1024-float vectors for every chunk is wasteful;
this single-element list is truthy (so ``_embed_missing`` skips
the chunk) without the memory cost.
"""

type _Row = dict[str, Any]


def _fetch_dicts(cursor: duckdb.DuckDBPyConnection) -> list[_Row]:
    """Convert cursor results to dicts keyed by column name.

    Standard DBAPI2 pattern — DuckDB has no built-in dict row factory.
    """
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row, strict=True)) for row in cursor.fetchall()]


def _row_to_chunk(row: _Row) -> Chunk:
    """Convert a dict row to a ``Chunk`` model."""
    raw_meta = str(row["metadata"]) if row["metadata"] else "{}"
    meta: ImportMeta = json.loads(raw_meta)

    has_emb = bool(row.get("has_embedding"))

    return Chunk(
        id=str(row["id"]),
        blob_sha=str(row["blob_sha"]),
        file_path=str(row["file_path"]),
        kind=ChunkKind(str(row["kind"])),
        name=str(row["name"]),
        scope=str(row["scope"]),
        content=str(row["content"]),
        line_start=int(row["line_start"]),
        line_end=int(row["line_end"]),
        metadata=meta,
        embedding=_EMBEDDING_SENTINEL if has_emb else [],
    )


# ── Store ────────────────────────────────────────────────────────────


class IndexStore:
    """DuckDB-backed storage for the code index.

    All public methods use ``_cur()`` which returns a fresh cursor
    for each operation.  DuckDB connections are **not** thread-safe,
    but cursors obtained via ``connection.cursor()`` are isolated
    per-call, so the indexing daemon thread and the main UI thread
    can safely share one ``IndexStore`` instance.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is not None:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(db_path) if db_path else ":memory:")
        self._local = threading.local()
        self._con.execute("INSTALL fts; LOAD fts;")
        for stmt in _SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._con.execute(stmt)
        self._fts_dirty = True

    def _cur(self) -> duckdb.DuckDBPyConnection:
        """Return a thread-local cursor — safe to call from any thread.

        Caches one cursor per thread to avoid the overhead of
        ``connection.cursor()`` on every operation (which triggers
        ``getcwd`` + ``stat`` syscalls in DuckDB).
        """
        cur = getattr(self._local, "cur", None)
        if cur is None:
            cur = self._con.cursor()
            self._local.cur = cur
        return cur

    def checkpoint(self) -> None:
        """Force-flush pending writes so other threads can read them.

        DuckDB's MVCC means concurrent writes from one thread block
        reads from other threads.  Call this after a batch of inserts
        (before starting a long-running update phase like embedding)
        to ensure reads from the main/UI thread see the data.
        """
        self._con.execute("CHECKPOINT")

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    # ── Writes ───────────────────────────────────────────────────────

    def insert_snapshot(self, commit_sha: str, file_path: str, blob_sha: str) -> None:
        """Record that *commit_sha* contains *file_path* at *blob_sha*."""
        self._cur().execute(_INSERT_SNAPSHOT_SQL, [commit_sha, file_path, blob_sha])

    def _bulk_insert(self, sql: str, table: pa.Table) -> None:
        """Register *table* as ``_stg``, execute *sql*, then unregister.

        All bulk-insert SQL files reference the fixed view name ``_stg``.
        Cursor-scoped registration means each thread's ``_stg`` is
        isolated — no naming conflicts.
        """
        cur = self._cur()
        cur.register("_stg", table)
        try:
            cur.execute(sql)
        finally:
            cur.unregister("_stg")

    def insert_snapshots(self, rows: list[tuple[str, str, str]]) -> None:
        """Batch insert snapshot rows: ``[(commit_sha, file_path, blob_sha), ...]``."""
        if not rows:
            return
        self._bulk_insert(_UPSERT_SNAPSHOTS_SQL, snapshots_to_table(rows))

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        """Batch insert chunks via DuckDB's columnar register API.

        Conversion to PyArrow is handled by :func:`chunks_to_table`.
        Embeddings are always ``NULL`` on initial insert — set later
        via :meth:`update_embedding`.
        """
        if not chunks:
            return
        self._bulk_insert(_UPSERT_CHUNKS_SQL, chunks_to_table(chunks))
        self._fts_dirty = True

    def delete_edges(self, commit_sha: str) -> None:
        """Remove all edges scoped to *commit_sha*."""
        self._cur().execute(_DELETE_EDGES_SQL, [commit_sha])

    def insert_edges(self, edges: list[Edge], commit_sha: str) -> None:
        """Batch insert edges scoped to *commit_sha*."""
        if not edges:
            return
        self._bulk_insert(_INSERT_EDGES_SQL, edges_to_table(edges, commit_sha))

    def update_embedding(self, chunk_id: str, embedding: list[float]) -> None:
        """Set the embedding vector for a single chunk."""
        self._cur().execute(_UPDATE_EMBEDDING_SQL, [embedding, chunk_id])

    def update_embeddings(self, ids: list[str], embeddings: list[list[float]]) -> None:
        """Batch-update embedding vectors via PyArrow join.

        Individual ``UPDATE ... WHERE id = ?`` costs ~67 ms/call due to
        DuckDB per-statement overhead.  A single join-UPDATE from a
        registered PyArrow table costs ~0.03 ms/row — **2000x faster**.
        """
        if not ids:
            return
        table = pa.table({"id": ids, "embedding": embeddings})
        cur = self._cur()
        cur.register("_emb_stg", table)
        try:
            cur.execute(_UPDATE_EMBEDDINGS_SQL)
        finally:
            cur.unregister("_emb_stg")

    def clear_embeddings(self) -> int:
        """Set all embeddings to NULL.

        Returns the number of chunks that had embeddings cleared.
        Used when switching embedding models — vectors from different
        models are not comparable.
        """
        row = self._cur().execute(_CLEAR_EMBEDDINGS_SQL).fetchone()
        return int(row[0]) if row else 0

    # ── Hygiene ──────────────────────────────────────────────────────

    def count_orphan_chunks(self) -> int:
        """Count chunks not referenced by any file snapshot."""
        row = self._cur().execute(_COUNT_ORPHAN_CHUNKS_SQL).fetchone()
        return int(row[0]) if row else 0

    def prune_orphans(self) -> tuple[int, int]:
        """Delete chunks and edges not referenced by any file snapshot.

        Returns ``(chunks_deleted, edges_deleted)``.
        """
        cur = self._cur()
        edge_row = cur.execute(_PRUNE_EDGES_SQL).fetchone()
        edges_deleted = int(edge_row[0]) if edge_row else 0
        chunk_row = cur.execute(_PRUNE_CHUNKS_SQL).fetchone()
        chunks_deleted = int(chunk_row[0]) if chunk_row else 0
        if chunks_deleted > 0:
            self._fts_dirty = True
        return chunks_deleted, edges_deleted

    # ── Reads ────────────────────────────────────────────────────────

    def has_blob(self, blob_sha: str) -> bool:
        """Check whether any chunks exist for *blob_sha*."""
        result = self._cur().execute(_HAS_BLOB_SQL, [blob_sha]).fetchone()
        return result is not None

    def get_chunks(
        self,
        commit_sha: str,
        *,
        file_path: str | None = None,
        kind: ChunkKind | None = None,
        name: str | None = None,
    ) -> list[Chunk]:
        """Query chunks visible at *commit_sha* with optional filters."""
        kind_val = kind.value if kind is not None else None
        params: list[str | None] = [
            commit_sha,
            file_path,
            file_path,
            kind_val,
            kind_val,
            name,
            name,
        ]
        rows = _fetch_dicts(self._cur().execute(_GET_CHUNKS_SQL, params))
        return [_row_to_chunk(r) for r in rows]

    def get_edges(
        self,
        commit_sha: str,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        kind: EdgeKind | None = None,
    ) -> list[Edge]:
        """Query edges scoped to *commit_sha* with optional filters."""
        kind_val = kind.value if kind is not None else None
        params: list[str | None] = [
            commit_sha,
            source_id,
            source_id,
            target_id,
            target_id,
            kind_val,
            kind_val,
        ]
        rows = _fetch_dicts(self._cur().execute(_GET_EDGES_SQL, params))
        return [
            Edge(
                source_id=str(r["source_id"]),
                target_id=str(r["target_id"]),
                kind=EdgeKind(str(r["kind"])),
            )
            for r in rows
        ]

    def diff_chunks(
        self,
        base_sha: str,
        head_sha: str,
    ) -> tuple[list[Chunk], list[Chunk], list[Chunk]]:
        """Compare chunks between two commits.

        Returns ``(added, removed, modified)`` where *modified* means
        the same file path exists in both but with a different blob SHA.
        """
        added = _fetch_dicts(self._cur().execute(_DIFF_ADDED_SQL, [head_sha, base_sha]))
        removed = _fetch_dicts(self._cur().execute(_DIFF_REMOVED_SQL, [base_sha, head_sha]))
        modified = _fetch_dicts(self._cur().execute(_DIFF_MODIFIED_SQL, [head_sha, base_sha]))
        return (
            [_row_to_chunk(r) for r in added],
            [_row_to_chunk(r) for r in removed],
            [_row_to_chunk(r) for r in modified],
        )

    def search_by_name(self, commit_sha: str, pattern: str) -> list[Chunk]:
        """Find chunks whose name contains *pattern* (case-insensitive)."""
        rows = _fetch_dicts(self._cur().execute(_SEARCH_BY_NAME_SQL, [commit_sha, f"%{pattern}%"]))
        return [_row_to_chunk(r) for r in rows]

    def search_similar(
        self,
        commit_sha: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Find the *top_k* chunks most similar to *query_embedding*.

        Uses DuckDB built-in ``list_cosine_similarity()``.
        """
        rows = _fetch_dicts(
            self._cur().execute(_SEARCH_SIMILAR_SQL, [query_embedding, commit_sha, top_k])
        )
        return [(_row_to_chunk(r), float(r["score"])) for r in rows]

    def search_by_text(
        self,
        commit_sha: str,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Semantic search: embed *query* then find similar chunks.

        Convenience wrapper around ``search_similar()`` that handles
        embedding the query text.
        """
        from rbtr.index.embeddings import embed_text  # deferred: heavy native lib

        query_embedding = embed_text(query)
        return self.search_similar(commit_sha, query_embedding, top_k)

    # ── FTS ──────────────────────────────────────────────────────────

    def rebuild_fts_index(self) -> None:
        """(Re)create the BM25 full-text search index on chunks."""
        with contextlib.suppress(duckdb.CatalogException):
            self._cur().execute("PRAGMA drop_fts_index('chunks');")
        self._cur().execute("PRAGMA create_fts_index('chunks', 'id', 'name', 'content');")
        self._fts_dirty = False

    def _ensure_fts(self) -> None:
        """Rebuild the FTS index if stale.

        DuckDB's FTS index is in-memory and not persisted to disk,
        so it must be rebuilt after opening an existing database or
        after inserting new chunks.
        """
        if self._fts_dirty:
            self.rebuild_fts_index()

    def search_fulltext(
        self,
        commit_sha: str,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """BM25 keyword search across chunk name and content.

        Automatically rebuilds the FTS index if it is stale or was
        lost (DuckDB FTS indexes are in-memory only).
        """
        self._ensure_fts()
        rows = _fetch_dicts(self._cur().execute(_SEARCH_FULLTEXT_SQL, [query, commit_sha, top_k]))
        return [(_row_to_chunk(r), float(r["score"])) for r in rows]
