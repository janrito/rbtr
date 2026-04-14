"""DuckDB storage — schema, read/write, and search for the code index.

The store manages three tables:

- `file_snapshots` maps a commit SHA to its file tree (path → blob SHA).
- `chunks` holds indexed content, keyed by blob SHA so identical files
  across commits are stored once.
- `edges` records relationships between chunks, scoped per commit.

All commit-scoped queries join through `file_snapshots` to resolve
which chunks belong to a given snapshot.
"""

from __future__ import annotations

import contextlib
import importlib.resources
import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]  # no stubs available

from rbtr_legacy.config import config
from rbtr_legacy.index.arrow import chunks_to_table, edges_to_table, snapshots_to_table
from rbtr_legacy.index.models import Chunk, ChunkKind, Edge, EdgeKind, ImportMeta

if TYPE_CHECKING:
    from rbtr_legacy.index.search import ScoredResult

log = logging.getLogger(__name__)

# Bump this when the schema changes.  On open, if the stored version
# doesn't match, the DB file is deleted and rebuilt from scratch.
SCHEMA_VERSION = 2

# Bump this when the embedding text format changes.  On open, if the
# stored version doesn't match, all embeddings are cleared so
# _embed_missing() re-computes them on the next index build.
EMBEDDING_VERSION = 1

# Import and doc_section chunks have short, keyword-dense content
# that produces misleadingly high cosine scores.  Filtering them
# from the semantic pool lets real function/class definitions surface.
_SEMANTIC_EXCLUDE = frozenset({ChunkKind.IMPORT, ChunkKind.DOC_SECTION})

# ── SQL loader ───────────────────────────────────────────────────────

_sql_pkg = importlib.resources.files("rbtr_legacy.index") / "sql"


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
_INBOUND_DEGREE_SQL = _load_sql("inbound_degree.sql")
_DELETE_SNAPSHOTS_SQL = _load_sql("delete_snapshots.sql")
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

`Chunk.embedding` is `list[float]` — empty means absent.
Loading full 1024-float vectors for every chunk is wasteful;
this single-element list is truthy (so `_embed_missing` skips
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
    """Convert a dict row to a `Chunk` model."""
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
        content_tokens=str(row.get("content_tokens", "")),
        name_tokens=str(row.get("name_tokens", "")),
        line_start=int(row["line_start"]),
        line_end=int(row["line_end"]),
        metadata=meta,
        embedding=_EMBEDDING_SENTINEL if has_emb else [],
    )


# ── Schema versioning ────────────────────────────────────────────────


def _check_schema_version(db_path: Path) -> None:
    """Delete *db_path* if its schema version doesn't match.

    Opens a temporary connection to read the `meta` table.  If the
    stored version is stale (or the table doesn't exist), the file is
    removed so the next `IndexStore.__init__` creates a fresh DB.
    """
    if not db_path.exists():
        return
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchall()
        except duckdb.CatalogException:
            # No meta table at all → old schema.
            rows = []
        con.close()
    except duckdb.ConnectionException:
        # Already open in this process with a different config
        # (e.g. read-write).  The existing connection validated
        # the schema on creation — safe to skip.
        return
    except duckdb.IOException:
        # Corrupt or locked — nuke it.
        rows = []

    stored = int(rows[0][0]) if rows else 0
    if stored != SCHEMA_VERSION:
        log.warning(
            "Index schema changed (v%d→v%d), rebuilding.",
            stored,
            SCHEMA_VERSION,
        )
        db_path.unlink(missing_ok=True)
        # DuckDB may create a .wal file alongside the DB.
        wal = db_path.with_suffix(db_path.suffix + ".wal")
        wal.unlink(missing_ok=True)


# ── Store ────────────────────────────────────────────────────────────


class IndexStore:
    """DuckDB-backed storage for the code index.

    All public methods use `_cur()` which returns a fresh cursor
    for each operation.  DuckDB connections are **not** thread-safe,
    but cursors obtained via `connection.cursor()` are isolated
    per-call, so the indexing daemon thread and the main UI thread
    can safely share one `IndexStore` instance.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is not None:
            resolved = Path(db_path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            _check_schema_version(resolved)
        self._con = duckdb.connect(str(db_path) if db_path else ":memory:")
        self._local = threading.local()
        self._con.execute("INSTALL fts; LOAD fts;")
        for stmt in _SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._con.execute(stmt)
        # Stamp the schema version so future opens can detect staleness.
        self._con.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            [str(SCHEMA_VERSION)],
        )
        # Check embedding version — clear embeddings if the text format
        # changed, so _embed_missing() re-computes them on next build.
        self._check_embedding_version()
        self._fts_dirty = True
        self._fts_lock = threading.Lock()

    def _check_embedding_version(self) -> None:
        """Clear embeddings if the text format or model changed."""
        rows = self._con.execute(
            "SELECT key, value FROM meta WHERE key IN ('embedding_version', 'embedding_model')"
        ).fetchall()
        stored = dict(rows)
        stored_version = int(stored.get("embedding_version", "0"))
        stored_model = stored.get("embedding_model", "")
        current_model = config.index.embedding_model

        version_changed = stored_version != EMBEDDING_VERSION
        model_changed = stored_model != current_model

        if version_changed or model_changed:
            n = self._con.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()
            count = n[0] if n else 0
            if count > 0:
                if version_changed:
                    reason = f"format v{stored_version}→v{EMBEDDING_VERSION}"
                else:
                    reason = f"model {stored_model!r}→{current_model!r}"
                log.warning(
                    "Embedding config changed (%s), clearing %d embeddings.",
                    reason,
                    count,
                )
                self._con.execute(_CLEAR_EMBEDDINGS_SQL)
            self._con.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES "
                "('embedding_version', ?), ('embedding_model', ?)",
                [str(EMBEDDING_VERSION), current_model],
            )

    def _cur(self) -> duckdb.DuckDBPyConnection:
        """Return a thread-local cursor — safe to call from any thread.

        Caches one cursor per thread to avoid the overhead of
        `connection.cursor()` on every operation (which triggers
        `getcwd` + `stat` syscalls in DuckDB).
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

    def delete_snapshots(self, commit_sha: str) -> None:
        """Remove all file snapshots scoped to *commit_sha*."""
        self._cur().execute(_DELETE_SNAPSHOTS_SQL, [commit_sha])

    def _bulk_insert(self, sql: str, table: pa.Table) -> None:
        """Register *table* as `_stg`, execute *sql*, then unregister.

        All bulk-insert SQL files reference the fixed view name `_stg`.
        Cursor-scoped registration means each thread's `_stg` is
        isolated — no naming conflicts.
        """
        cur = self._cur()
        cur.register("_stg", table)
        try:
            cur.execute(sql)
        finally:
            cur.unregister("_stg")

    def insert_snapshots(self, rows: list[tuple[str, str, str]]) -> None:
        """Batch insert snapshot rows: `[(commit_sha, file_path, blob_sha), ...]`."""
        if not rows:
            return
        self._bulk_insert(_UPSERT_SNAPSHOTS_SQL, snapshots_to_table(rows))

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        """Batch insert chunks via DuckDB's columnar register API.

        Conversion to PyArrow is handled by `chunks_to_table`.
        Embeddings are always `NULL` on initial insert — set later
        via `update_embedding`.
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

        Individual `UPDATE ... WHERE id = ?` costs ~67 ms/call due to
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

        Returns `(chunks_deleted, edges_deleted)`.
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

    def inbound_degrees(
        self,
        commit_sha: str,
        chunk_ids: list[str],
    ) -> dict[str, int]:
        """Return inbound edge counts for the given chunk IDs."""
        if not chunk_ids:
            return {}
        rows = _fetch_dicts(self._cur().execute(_INBOUND_DEGREE_SQL, [commit_sha, chunk_ids]))
        return {str(r["chunk_id"]): int(r["degree"]) for r in rows}

    def diff_chunks(
        self,
        base_sha: str,
        head_sha: str,
    ) -> tuple[list[Chunk], list[Chunk], list[Chunk]]:
        """Compare chunks between two commits.

        Returns `(added, removed, modified)` where *modified* means
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

        Uses DuckDB built-in `list_cosine_similarity()`.
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

        Convenience wrapper around `search_similar()` that handles
        embedding the query text.
        """
        from rbtr_legacy.index.embeddings import embed_text  # deferred: heavy native lib

        query_embedding = embed_text(query)
        return self.search_similar(commit_sha, query_embedding, top_k)

    # ── FTS ──────────────────────────────────────────────────────────

    def rebuild_fts_index(self) -> None:
        """(Re)create the BM25 full-text search index on chunks.

        Indexes the pre-tokenised `name_tokens` and
        `content_tokens` columns with stemming and stopwords
        disabled — essential for code search where identifiers
        must survive intact.

        After building the index, IDF is neutralised by setting
        every term's document frequency to 1.  This prevents
        common-but-important terms like `config` or `model`
        from being suppressed, reducing BM25 to TF + length
        normalisation.
        """
        cur = self._cur()
        with contextlib.suppress(duckdb.CatalogException):
            cur.execute("PRAGMA drop_fts_index('chunks');")
        cur.execute(
            "PRAGMA create_fts_index("
            "  'chunks', 'id', 'name_tokens', 'content_tokens',"
            "  stemmer='none', stopwords='none',"
            "  ignore='([^a-z0-9_])+'"
            ");"
        )
        # Neutralise IDF: every term gets df=1 so BM25 scores
        # depend only on term frequency and document length.
        cur.execute("UPDATE fts_main_chunks.dict SET df = 1")
        self._fts_dirty = False

    def _ensure_fts(self) -> None:
        """Rebuild the FTS index if stale.

        DuckDB's FTS index is in-memory and not persisted to disk,
        so it must be rebuilt after opening an existing database or
        after inserting new chunks.

        Serialised with a lock because the DDL statements
        (`drop_fts_index` / `create_fts_index`) trigger
        catalog writes that conflict under DuckDB's MVCC when
        executed concurrently from different cursors.
        """
        if self._fts_dirty:
            with self._fts_lock:
                # Double-check: another thread may have rebuilt while
                # we waited for the lock.
                if self._fts_dirty:
                    self.rebuild_fts_index()

    def search_fulltext(
        self,
        commit_sha: str,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """BM25 keyword search across chunk name and content.

        The *query* is pre-tokenised with `tokenise_code` so
        that identifier queries (`AgentDeps` → `agentdeps agent
        deps`) match the code-aware tokens stored in the index.

        Automatically rebuilds the FTS index if it is stale or was
        lost (DuckDB FTS indexes are in-memory only).
        """
        from rbtr_legacy.index.tokenise import tokenise_code

        self._ensure_fts()
        tokenised_query = tokenise_code(query)
        if not tokenised_query:
            return []
        rows = _fetch_dicts(
            self._cur().execute(_SEARCH_FULLTEXT_SQL, [tokenised_query, commit_sha, top_k])
        )
        return [(_row_to_chunk(r), float(r["score"])) for r in rows]

    # ── Unified search ───────────────────────────────────────────────

    def search(
        self,
        commit_sha: str,
        query: str,
        *,
        top_k: int = 10,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        changed_files: set[str] | None = None,
    ) -> list[ScoredResult]:
        """Unified search fusing lexical, semantic, and name signals.

        Runs three retrieval channels, normalises and fuses their
        scores, then applies kind-boost, file-category, importance
        (inbound-degree), and proximity (diff distance) multipliers.
        Returns up to *top_k* `ScoredResult` objects with
        full signal breakdown.

        Falls back gracefully: if embeddings are unavailable, the
        semantic channel is skipped (its weight is redistributed
        to the other channels).

        Args:
            changed_files: Set of file paths in the current diff.
                When provided, chunks near changed files get a
                proximity boost.
        """
        from rbtr_legacy.index.search import (
            fuse_scores,
            importance_score,
            name_score,
            proximity_score,
            weights_for_query,
        )

        if alpha is not None and beta is not None and gamma is not None:
            a, b, g = alpha, beta, gamma
        else:
            a, b, g = weights_for_query(query)

        # ── Channel 1: BM25 lexical ─────────────────────────────
        # Fetch more than top_k so fusion has a good candidate pool.
        pool_size = top_k * 5
        lexical_results = self.search_fulltext(commit_sha, query, top_k=pool_size)
        lexical_scores: dict[str, float] = {}
        candidates: dict[str, Chunk] = {}
        for chunk, score in lexical_results:
            lexical_scores[chunk.id] = score
            candidates[chunk.id] = chunk

        # ── Channel 2: semantic (embedding cosine) ───────────────
        semantic_scores: dict[str, float] = {}
        try:
            sem_fetch = pool_size * 2  # over-fetch to compensate
            semantic_results = self.search_by_text(commit_sha, query, top_k=sem_fetch)
            for chunk, score in semantic_results:
                if chunk.kind in _SEMANTIC_EXCLUDE:
                    continue
                semantic_scores[chunk.id] = score
                if chunk.id not in candidates:
                    candidates[chunk.id] = chunk
                if len(semantic_scores) >= pool_size:
                    break
        except Exception:
            # Embeddings unavailable — redistribute weight.
            if a > 0:
                b_new = b + a * (b / (b + g)) if (b + g) > 0 else b + a / 2
                g_new = g + a * (g / (b + g)) if (b + g) > 0 else g + a / 2
                a, b, g = 0.0, b_new, g_new

        # ── Channel 3: name match ───────────────────────────────
        # Score all candidates plus any additional name matches.
        # For multi-word queries, also search individual tokens
        # so that token-level matches enter the candidate pool
        # (e.g. "import edge" → "infer_import_edges").
        name_matches = self.search_by_name(commit_sha, query)
        for chunk in name_matches:
            if chunk.id not in candidates:
                candidates[chunk.id] = chunk

        tokens = query.split()
        if len(tokens) > 1:
            for token in tokens:
                if len(token) >= 3:
                    for chunk in self.search_by_name(commit_sha, token):
                        if chunk.id not in candidates:
                            candidates[chunk.id] = chunk

        name_scores_map: dict[str, float] = {}
        for cid, chunk in candidates.items():
            ns = name_score(query, chunk.name)
            if ns > 0.0:
                name_scores_map[cid] = ns

        # ── Importance (inbound-degree) ──────────────────────
        candidate_ids = list(candidates.keys())
        degrees = self.inbound_degrees(commit_sha, candidate_ids)
        importance_map: dict[str, float] = {
            cid: importance_score(degrees.get(cid, 0)) for cid in candidate_ids
        }

        # ── Proximity (diff distance) ────────────────────────
        proximity_map: dict[str, float] | None = None
        if changed_files:
            # Collect IDs of candidate-adjacent chunks via edges.
            # Then check if any neighbour lives in a changed file.
            candidate_set = set(candidate_ids)
            all_edges = self.get_edges(commit_sha)

            # Neighbours: for each candidate, collect IDs of chunks
            # on the other end of an edge.
            neighbours: dict[str, set[str]] = {cid: set() for cid in candidate_set}
            for e in all_edges:
                if e.source_id in candidate_set:
                    neighbours[e.source_id].add(e.target_id)
                if e.target_id in candidate_set:
                    neighbours[e.target_id].add(e.source_id)

            # Resolve neighbour IDs to file paths.  Candidates
            # are already loaded; non-candidate neighbours need
            # a file_path lookup.
            all_neighbour_ids = set()
            for nbs in neighbours.values():
                all_neighbour_ids |= nbs
            unknown_ids = all_neighbour_ids - candidate_set
            neighbour_paths: dict[str, str] = {cid: c.file_path for cid, c in candidates.items()}
            if unknown_ids:
                rows = _fetch_dicts(
                    self._cur().execute(
                        "SELECT id, file_path FROM chunks WHERE id IN (SELECT unnest(?::text[]))",
                        [list(unknown_ids)],
                    )
                )
                for r in rows:
                    neighbour_paths[str(r["id"])] = str(r["file_path"])

            # A candidate "has edge to changed" if any neighbour
            # is in a changed file.
            has_edge: set[str] = set()
            for cid, nbs in neighbours.items():
                for nb in nbs:
                    nb_path = neighbour_paths.get(nb, "")
                    if nb_path in changed_files:
                        has_edge.add(cid)
                        break

            proximity_map = {
                cid: proximity_score(
                    chunk.file_path,
                    changed_files,
                    has_edge_to_changed=cid in has_edge,
                )
                for cid, chunk in candidates.items()
            }

        return fuse_scores(
            candidates,
            lexical_scores,
            semantic_scores,
            name_scores_map,
            alpha=a,
            beta=b,
            gamma=g,
            top_k=top_k,
            importance_scores=importance_map,
            proximity_scores=proximity_map,
        )
