"""Write session — transactional write surface for the index.

All mutations go through `WriteSession`, obtained via
`IndexStore.session()`.  The session is a context manager:

- `__enter__`: begins a transaction.
- `__exit__`: flushes buffered chunks, commits the
  transaction, and rebuilds the FTS index if chunks were
  modified.  Rolls back on exception.

Sweep is explicit: call `ws.sweep()` to remove residue from
crashed builds.

`IndexStore` (in `store.py`) owns the connection, reads,
and search.  `WriteSession` owns all data mutations.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

import duckdb
import polars as pl
import structlog

from rbtr.config import config
from rbtr.index import load_sql
from rbtr.index.constants import EMBEDDING_FORMAT_VERSION, SCHEMA_VERSION
from rbtr.index.frames import chunks_frame, edges_frame, embeddings_frame, snapshots_frame
from rbtr.index.models import Edge, GcCounts, Snapshot, TokenisedChunk

_DELETE_CHUNKS_FOR_BLOBS_SQL = load_sql("delete_chunks_for_blobs.sql")
_ADD_WATCHED_REFS_SQL = load_sql("insert_watched_refs.sql")
_REMOVE_WATCHED_REFS_SQL = load_sql("delete_watched_refs.sql")
_DELETE_EDGES_SQL = load_sql("delete_edges.sql")
_DELETE_SNAPSHOTS_SQL = load_sql("delete_snapshots.sql")
_DROP_COMMIT_SQL = load_sql("drop_commit.sql")
_HAS_ANY_INDEXED_SQL = load_sql("has_any_indexed.sql")
_INSERT_EDGES_SQL = load_sql("insert_edges.sql")
_MARK_INDEXED_SQL = load_sql("mark_indexed.sql")
_ORPHAN_REPO_IDS_SQL = load_sql("orphan_repo_ids.sql")
_PRUNE_CHUNKS_SQL = load_sql("prune_chunks.sql")
_PRUNE_EDGES_SQL = load_sql("prune_edges.sql")
_SWEEP_ORPHAN_CHUNKS_SQL = load_sql("sweep_orphan_chunks.sql")
_SWEEP_ORPHAN_EDGES_SQL = load_sql("sweep_orphan_edges.sql")
_SWEEP_ORPHAN_SNAPSHOTS_SQL = load_sql("sweep_orphan_snapshots.sql")
_UPDATE_EMBEDDINGS_SQL = load_sql("update_embeddings.sql")
_UPSERT_CHUNKS_SQL = load_sql("upsert_chunks.sql")
_UPSERT_SNAPSHOTS_SQL = load_sql("upsert_snapshots.sql")
_REGISTER_REPO_SQL = load_sql("register_repo.sql")
_CREATE_FTS_INDEX_SQL = load_sql("create_fts_index.sql")
_NEUTRALISE_FTS_IDF_SQL = load_sql("neutralise_fts_idf.sql")
_SCHEMA_SQL = load_sql("schema.sql")
_SET_SCHEMA_VERSION_SQL = load_sql("set_schema_version.sql")
_GET_EMBEDDING_META_SQL = load_sql("get_embedding_meta.sql")
_COUNT_EMBEDDINGS_SQL = load_sql("count_embeddings.sql")
_CLEAR_ALL_EMBEDDINGS_SQL = load_sql("clear_all_embeddings.sql")
_SET_EMBEDDING_VERSION_SQL = load_sql("set_embedding_version.sql")
_SET_EMBEDDING_MODEL_SQL = load_sql("set_embedding_model.sql")

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from rbtr.index.store import IndexStore


class WriteSession:
    """Transactional write surface for the index store.

    Use as a context manager via `IndexStore.session()`:

    ```python
    with store.session() as ws:
        ws.add_chunk(chunk)   # buffered (chunk carries its repo_id)
        ws.insert_edges(...)  # immediate
    # exit: flush buffer + commit + FTS rebuild
    ```

    All write methods require an active transaction and raise
    `RuntimeError` if called outside a session.  Chunks are
    buffered via `add_chunk` and flushed automatically when
    the buffer fills or before any operation that depends on
    chunks being in the DB.  Each chunk carries its own `repo_id`,
    so a single session may safely add chunks for more than one
    repo — they stay correctly attributed in the flush.

    **Concurrency warning:** in the daemon, `WriteSession`
    instances must only be created on the job worker thread
    (serialised by `_write_sem`).  The watcher, RPC handlers,
    and embed preemption checks must be read-only.
    """

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        self._active = False
        self._chunks_modified = False
        # Each chunk carries its own repo_id, so one flush may span
        # repos — the staging frame and upsert SQL read repo_id per
        # row.
        self._chunk_buffer: list[TokenisedChunk] = []

    @property
    def _cursor(self) -> duckdb.DuckDBPyConnection:
        """Thread-local cursor from the underlying store."""
        return self._store._cursor

    # ── Context manager ──────────────────────────────────────────

    def __enter__(self) -> WriteSession:
        if not self._store._bootstrapped:
            self._bootstrap()
        self._cursor.begin()
        self._active = True
        return self

    def _bootstrap(self) -> None:
        """Create schema and check embedding version on first use."""
        self._cursor.execute(_SCHEMA_SQL)
        self._cursor.execute(
            _SET_SCHEMA_VERSION_SQL,
            {"schema_version": str(SCHEMA_VERSION)},
        )
        self._check_embedding_version()
        self._store._bootstrapped = True

    def _check_embedding_version(self) -> None:
        """Clear embeddings if the format version or model changed."""
        rows = self._cursor.execute(_GET_EMBEDDING_META_SQL).fetchall()
        stored: dict[str, str] = {str(r[0]): str(r[1]) for r in rows}

        stored_version = stored.get("embedding_version", "")
        stored_model = stored.get("embedding_model", "")
        current_version = str(EMBEDDING_FORMAT_VERSION)
        current_model = config.embedding_model

        if stored_version == current_version and stored_model == current_model:
            return

        n = self._cursor.execute(_COUNT_EMBEDDINGS_SQL).fetchone()
        count = n[0] if n else 0
        if count > 0:
            log.warning(
                "embedding_config_changed",
                stored_model=stored_model or "<none>",
                current_model=current_model,
                stored_version=stored_version or "<none>",
                current_version=current_version,
                cleared=count,
            )
            self._cursor.execute(_CLEAR_ALL_EMBEDDINGS_SQL)
        self._cursor.execute(
            _SET_EMBEDDING_VERSION_SQL,
            {"embedding_version": current_version},
        )
        self._cursor.execute(
            _SET_EMBEDDING_MODEL_SQL,
            {"embedding_model": current_model},
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if not self._active:
            return
        if exc_type is not None:
            self._cursor.rollback()
            self._active = False
        else:
            self._commit()

    def _commit(self) -> None:
        """Flush pending chunks, commit, and rebuild FTS if needed."""
        self._require_active()
        self._flush_chunks()
        self._cursor.commit()
        if self._chunks_modified:
            self._rebuild_fts()
        self._active = False

    def _rebuild_fts(self) -> None:
        """(Re)create the BM25 index and neutralise IDF."""
        self._cursor.execute(_CREATE_FTS_INDEX_SQL)
        self._cursor.execute(_NEUTRALISE_FTS_IDF_SQL)

    def _require_active(self) -> None:
        if not self._active:
            msg = "No active transaction. Use 'with store.session() as ws:'"
            raise RuntimeError(msg)

    # ── Bulk helpers ─────────────────────────────────────────────

    def _bulk_insert(self, sql: str, frame: pl.DataFrame) -> None:
        """Register *frame* as `_stg`, execute *sql*, then unregister.

        Struct columns are encoded to JSON strings before
        registration so DuckDB sees TEXT matching the schema.
        """
        self._require_active()
        struct_cols = [
            c
            for c, d in zip(frame.columns, frame.dtypes, strict=True)
            if d.base_type() == pl.Struct
        ]
        if struct_cols:
            frame = frame.with_columns(pl.col(c).struct.json_encode() for c in struct_cols)
        self._cursor.register("_stg", frame)
        try:
            self._cursor.execute(sql)
        finally:
            self._cursor.unregister("_stg")

    # ── Write methods ────────────────────────────────────────────

    def register_repo(self, path: str) -> int:
        """Register a repo path and return its integer ID.

        Idempotent — returns the existing ID if already registered.
        """
        self._require_active()
        existing = self._store.get_repo_id(path)
        if existing is not None:
            return existing
        rows = self._cursor.execute(_REGISTER_REPO_SQL, {"path": path}).fetchall()
        return int(rows[0][0])

    def add_chunk(self, chunk: TokenisedChunk) -> None:
        """Buffer a chunk for insertion under its own `repo_id`.

        Chunks are accumulated and flushed to DuckDB in batches —
        on commit/exit, when the buffer reaches
        `config.insert_batch_size`, or before any operation that
        depends on chunks being in the DB.  Each chunk carries its
        `repo_id`, so a single session may add chunks for several
        repos and they stay correctly attributed.
        """
        self._require_active()
        self._chunk_buffer.append(chunk)
        if len(self._chunk_buffer) >= config.insert_batch_size:
            self._flush_chunks()

    def _flush_chunks(self) -> None:
        """Write buffered chunks to DuckDB in one batch."""
        if not self._chunk_buffer:
            return
        self._bulk_insert(_UPSERT_CHUNKS_SQL, chunks_frame(self._chunk_buffer))
        self._chunks_modified = True
        self._chunk_buffer.clear()

    def delete_chunks_for_blobs(self, blob_shas: set[str], repo_id: int) -> None:
        """Delete all chunks for the given blob SHAs.

        Called before re-inserting chunks when the detected
        language changed (e.g. a new plugin was registered).
        """
        if not blob_shas:
            return
        self._require_active()
        self._cursor.execute(
            _DELETE_CHUNKS_FOR_BLOBS_SQL, {"repo_id": repo_id, "blob_shas": list(blob_shas)}
        )
        self._chunks_modified = True

    def add_watched_refs(self, repo_id: int, refs: list[str]) -> None:
        """Add *refs* to the repo's watch set (idempotent on the PK)."""
        if not refs:
            return
        self._require_active()
        self._cursor.execute(_ADD_WATCHED_REFS_SQL, {"repo_id": repo_id, "refs": refs})

    def remove_watched_refs(self, repo_id: int, refs: list[str]) -> None:
        """Remove *refs* from the repo's watch set."""
        if not refs:
            return
        self._require_active()
        self._cursor.execute(_REMOVE_WATCHED_REFS_SQL, {"repo_id": repo_id, "refs": refs})

    def insert_snapshots(self, snapshots: list[Snapshot], repo_id: int) -> None:
        """Batch insert snapshots."""
        if not snapshots:
            return
        self._bulk_insert(_UPSERT_SNAPSHOTS_SQL, snapshots_frame(snapshots, repo_id))

    def replace_snapshots(self, commit_sha: str, snapshots: list[Snapshot], repo_id: int) -> None:
        """Atomically replace all snapshots for *commit_sha*."""
        self._flush_chunks()
        self.delete_snapshots(commit_sha, repo_id=repo_id)
        self.insert_snapshots(snapshots, repo_id=repo_id)

    def insert_edges(self, edges: list[Edge], commit_sha: str, repo_id: int) -> None:
        """Batch insert edges scoped to *commit_sha*."""
        if not edges:
            return
        self._bulk_insert(_INSERT_EDGES_SQL, edges_frame(edges, commit_sha, repo_id))

    def replace_edges(self, commit_sha: str, edges: list[Edge], repo_id: int) -> None:
        """Atomically replace all edges for *commit_sha*."""
        self.delete_edges(commit_sha, repo_id=repo_id)
        self.insert_edges(edges, commit_sha, repo_id=repo_id)

    def delete_snapshots(self, commit_sha: str, repo_id: int) -> None:
        """Remove all file snapshots scoped to *commit_sha*."""
        self._require_active()
        self._cursor.execute(_DELETE_SNAPSHOTS_SQL, {"repo_id": repo_id, "commit_sha": commit_sha})

    def delete_edges(self, commit_sha: str, repo_id: int) -> None:
        """Remove edges scoped to *commit_sha*."""
        self._require_active()
        self._cursor.execute(_DELETE_EDGES_SQL, {"repo_id": repo_id, "commit_sha": commit_sha})

    def update_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        repo_id: int,
        truncated: list[bool] | None = None,
    ) -> None:
        """Batch-update embedding vectors via a polars staging frame."""
        if not ids:
            return
        self._require_active()
        self._flush_chunks()
        if truncated is None:
            truncated = [False] * len(ids)
        frame = embeddings_frame(ids, embeddings, truncated)
        self._cursor.register("_emb_stg", frame)
        try:
            self._cursor.execute(_UPDATE_EMBEDDINGS_SQL, {"repo_id": repo_id})
        finally:
            self._cursor.unregister("_emb_stg")

    def mark_indexed(self, repo_id: int, commit_sha: str) -> None:
        """Record a commit as fully indexed."""
        self._require_active()
        self._cursor.execute(_MARK_INDEXED_SQL, {"repo_id": repo_id, "commit_sha": commit_sha})

    # ── GC ───────────────────────────────────────────────────────

    def drop_commit(self, repo_id: int, commit_sha: str) -> GcCounts:
        """Remove all trace of *commit_sha* from this repo."""
        self._require_active()
        commit_row = self._cursor.execute(
            _DROP_COMMIT_SQL, {"repo_id": repo_id, "commit_sha": commit_sha}
        ).fetchone()
        snap_row = self._cursor.execute(
            _DELETE_SNAPSHOTS_SQL, {"repo_id": repo_id, "commit_sha": commit_sha}
        ).fetchone()
        edge_row = self._cursor.execute(
            _DELETE_EDGES_SQL, {"repo_id": repo_id, "commit_sha": commit_sha}
        ).fetchone()
        chunk_row = self._cursor.execute(_SWEEP_ORPHAN_CHUNKS_SQL, {"repo_id": repo_id}).fetchone()
        chunks_deleted = int(chunk_row[0]) if chunk_row else 0
        if chunks_deleted > 0:
            self._chunks_modified = True
        return GcCounts(
            commits=int(commit_row[0]) if commit_row else 0,
            snapshots=int(snap_row[0]) if snap_row else 0,
            edges=int(edge_row[0]) if edge_row else 0,
            chunks=chunks_deleted,
        )

    def cleanup(self, repo_id: int) -> GcCounts:
        """Remove all orphaned data for a repo.

        Combines crash-residue cleanup and stale-data pruning
        into a single pass:

        1. Delete snapshots/edges for commits never marked
           indexed (crash residue).
        2. Delete chunks not referenced by any surviving
           snapshot (stale extractions and crash residue).
        3. Delete edges whose commit has no snapshots.

        Returns counts of rows removed.
        """
        self._require_active()
        snap_row = self._cursor.execute(
            _SWEEP_ORPHAN_SNAPSHOTS_SQL, {"repo_id": repo_id}
        ).fetchone()
        orphan_edge_row = self._cursor.execute(
            _SWEEP_ORPHAN_EDGES_SQL, {"repo_id": repo_id}
        ).fetchone()
        chunk_row = self._cursor.execute(_PRUNE_CHUNKS_SQL, {"repo_id": repo_id}).fetchone()
        stale_edge_row = self._cursor.execute(_PRUNE_EDGES_SQL, {"repo_id": repo_id}).fetchone()
        chunks_deleted = int(chunk_row[0]) if chunk_row else 0
        if chunks_deleted > 0:
            self._chunks_modified = True
        return GcCounts(
            snapshots=int(snap_row[0]) if snap_row else 0,
            edges=(
                (int(orphan_edge_row[0]) if orphan_edge_row else 0)
                + (int(stale_edge_row[0]) if stale_edge_row else 0)
            ),
            chunks=chunks_deleted,
        )

    # ── Sweep ────────────────────────────────────────────────────

    def sweep(self) -> None:
        """Remove residue from builds that never completed.

        Only runs cleanup for repos that have at least one
        completed build.  A repo with zero indexed commits is
        mid-first-build, not crash residue.
        """
        self._require_active()
        rows = self._cursor.execute(_ORPHAN_REPO_IDS_SQL).fetchall()
        for (repo_id,) in rows:
            has_any = self._cursor.execute(_HAS_ANY_INDEXED_SQL, {"repo_id": repo_id}).fetchone()
            if has_any is None:
                continue
            self.cleanup(repo_id)
            log.info("swept_orphans", repo_id=repo_id)
