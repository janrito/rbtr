"""The index's write session: every change to a project's data goes through `WriteSession`."""

from __future__ import annotations

from types import TracebackType

import duckdb
import polars as pl
import structlog

from rbtr.config import config
from rbtr.domain.models import Edge, FileSnapshot, GcCounts, Repo
from rbtr.index import load_sql
from rbtr.index.db import build_fts, registered_views
from rbtr.index.staging import (
    TokenisedChunk,
    serial_map_view,
    staged_chunks,
    staged_edges,
    staged_embeddings,
    staged_file_snapshots,
)

_BLOB_IS_CURRENT_SQL = load_sql("blob_is_current.sql")
_COUNT_UNCLAIMED_BLOBS_SQL = load_sql("count_unclaimed_blobs.sql")
_GET_SNAPSHOT_LANGUAGE_SQL = load_sql("get_snapshot_language.sql")
_DELETE_CHUNKS_FOR_BLOBS_SQL = load_sql("delete_chunks_for_blobs.sql")
_ADD_WATCHED_REFS_SQL = load_sql("insert_watched_refs.sql")
_REMOVE_WATCHED_REFS_SQL = load_sql("delete_watched_refs.sql")
_DELETE_EDGES_SQL = load_sql("delete_edges.sql")
_DELETE_SNAPSHOTS_SQL = load_sql("delete_snapshots.sql")
_DROP_SNAPSHOT_SQL = load_sql("drop_snapshot.sql")
_INSERT_EDGES_SQL = load_sql("insert_edges.sql")
_MARK_INDEXED_SQL = load_sql("mark_indexed.sql")
_PRUNE_CHUNKS_SQL = load_sql("prune_chunks.sql")
_PRUNE_EDGES_SQL = load_sql("prune_edges.sql")
_SWEEP_ORPHAN_CHUNKS_SQL = load_sql("sweep_orphan_chunks.sql")
_SWEEP_ORPHAN_EDGES_SQL = load_sql("sweep_orphan_edges.sql")
_SWEEP_ORPHAN_SNAPSHOTS_SQL = load_sql("sweep_orphan_snapshots.sql")
_UPDATE_EMBEDDINGS_SQL = load_sql("update_embeddings.sql")
_UPSERT_CHUNKS_SQL = load_sql("upsert_chunks.sql")
_UPSERT_SNAPSHOTS_SQL = load_sql("upsert_snapshots.sql")

log = structlog.get_logger(__name__)


class WriteSession:
    """One transaction on one project's file, writing for one of its repos.

    Use as a context manager via `Index.session(repo)`:

    ```python
    with index.session(repo) as ws:
        ws.add_chunk(chunk)                        # buffered, content-addressed
        ws.insert_edges(edges, snapshot_sha=sha)   # immediate, under the repo
    # exit: flush buffer + commit + FTS rebuild; rollback on exception
    ```

    Every row a write adds belongs to the session's repo, so no method
    takes one.  Chunks are the project's: its checkouts share them.  The
    reads the transaction depends on (`blob_is_current`,
    `get_snapshot_language`) run on the session's cursor, so they see
    what it has written.

    Every method raises `RuntimeError` outside the `with` block.  The
    commit rolls back and raises `RuntimeError` when the session stored
    chunks for a `(blob_sha, file_language)` that no `file_snapshots`
    row in the file claims, so a snapshot's chunks and its rows are
    written in one transaction.  ARCHITECTURE's "Garbage collection"
    explains why.
    """

    def __init__(self, cursor: duckdb.DuckDBPyConnection, repo: Repo) -> None:
        self._cursor = cursor
        self._repo = repo
        self._active = False
        self._chunks_modified = False
        # One entry per (blob_sha, file_language) this session inserted
        # chunks for, checked against the claims at commit.
        self._inserted_blobs: set[tuple[str, str]] = set()
        # Chunks are content-addressed (keyed by id, no repo_id); the
        # buffer holds repo-independent content flushed via upsert.
        self._chunk_buffer: list[TokenisedChunk] = []

    def __enter__(self) -> WriteSession:
        self._cursor.begin()
        self._active = True
        return self

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
        """Flush pending chunks, commit, and rebuild FTS if chunks changed.

        Rolls back and raises `RuntimeError` when the session inserted
        chunks and wrote no `file_snapshots` row, rather than
        committing chunks the next clean-up would delete.  Rolls back
        and re-raises when the flush or the check fails.
        """
        self._require_active()
        try:
            self._flush_chunks()
            unclaimed = self._count_unclaimed_blobs()
        except BaseException:
            self._cursor.rollback()
            self._active = False
            raise
        if unclaimed > 0:
            self._cursor.rollback()
            self._active = False
            msg = (
                f"Session inserted chunks for {unclaimed} blob(s) with no "
                "file_snapshots row: a chunk no repo claims is deleted by "
                "the next clean-up."
            )
            raise RuntimeError(msg)
        self._cursor.commit()
        if self._chunks_modified:
            build_fts(self._cursor)
        self._active = False

    def _count_unclaimed_blobs(self) -> int:
        """How many inserted blobs no `file_snapshots` row claims."""
        if not self._inserted_blobs:
            return 0
        pairs = pl.DataFrame(
            sorted(self._inserted_blobs), schema=["blob_sha", "file_language"], orient="row"
        )
        with registered_views(self._cursor, _stg=pairs) as cur:
            return int(cur.sql(_COUNT_UNCLAIMED_BLOBS_SQL).fetchall()[0][0])

    def _require_active(self) -> None:
        if not self._active:
            msg = "No active transaction. Use 'with index.session(repo) as ws:'"
            raise RuntimeError(msg)

    def _bulk_insert(self, sql: str, frame: pl.DataFrame) -> None:
        """Execute *sql* against *frame*, bound as the view `_stg`.

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
        with registered_views(self._cursor, _stg=frame) as cur:
            cur.execute(sql)

    def blob_is_current(self, blob_sha: str, language: str, serials: dict[str, int]) -> bool:
        """Check whether *blob_sha* is up to date for host *language*.

        This is the blob-dedup check: the build calls it before extracting
        a file.  If True, the blob's chunks are current and valid — skip.
        If False, the file needs (re-)extraction.  The check spans the
        session's project (its chunks are content-addressed and shared by
        its checkouts), so a blob already parsed by any of them is reused —
        no re-parse when a linked worktree indexes it.  It runs on the
        session's cursor, so it sees chunks this transaction deleted.

        *language* is the file's currently detected host language;
        *serials* maps language id → current extraction serial (the
        full registry, plus `""` for plaintext).  The blob is up to
        date iff it has ≥1 chunk in *language* **and** every chunk's
        `(language, extraction_serial)` matches a row in
        *serials*.  Either condition failing triggers re-extraction:

        - Detected language changed — a blob indexed as plaintext
          (`language=""`), then a plugin registered for the
          extension → no chunk in the new language → re-extract.
          (Every file leaves a host-language chunk, so this check
          is reliable.)
        - A language's extraction serial is bumped → chunks stored at
          the old serial no longer match → re-extracted.
        - A multi-language file (SFC) lists every embedded language
          plus the host, so bumping *any* contributor (the svelte
          host or the delegated typescript) re-extracts the file;
          when none change, it is skipped like any other blob.

        When `blob_is_current` returns False and old chunks exist for
        the blob (language change), the caller must delete old
        chunks before inserting new ones — the new extraction
        may produce different chunk IDs that the upsert can't
        reconcile.
        """
        self._require_active()
        with registered_views(self._cursor, _serial_map=serial_map_view(serials)) as cur:
            rows = cur.sql(
                _BLOB_IS_CURRENT_SQL, params={"blob_sha": blob_sha, "language": language}
            ).fetchall()
        return bool(rows[0][0])

    def get_snapshot_language(self, file_path: str) -> str:
        """Return the language *file_path* was detected as in any of the repo's snapshots.

        Returns `''` if the file has never been indexed or was
        indexed as plaintext.
        """
        self._require_active()
        rows = self._cursor.sql(
            _GET_SNAPSHOT_LANGUAGE_SQL,
            params={"repo_id": self._repo.repo_id, "file_path": file_path},
        ).fetchall()
        return str(rows[0][0]) if rows else ""

    def add_chunk(self, chunk: TokenisedChunk) -> None:
        """Buffer a content-addressed chunk for insertion.

        Chunks are accumulated and flushed to DuckDB in batches —
        on commit/exit, when the buffer reaches
        `config.insert_batch_size`, or before any operation that
        depends on chunks being in the DB.  Chunks carry no `repo_id`
        (they are keyed by content hash `id` and shared by the project's
        checkouts);
        a repo claims a chunk by inserting a `file_snapshots` row for
        its blob in the same transaction, which `_commit` requires.
        """
        self._require_active()
        self._chunk_buffer.append(chunk)
        if len(self._chunk_buffer) >= config.insert_batch_size:
            self._flush_chunks()

    def _flush_chunks(self) -> None:
        """Write buffered chunks to DuckDB in one batch."""
        if not self._chunk_buffer:
            return
        self._bulk_insert(_UPSERT_CHUNKS_SQL, staged_chunks(self._chunk_buffer))
        self._chunks_modified = True
        self._inserted_blobs.update((c.blob_sha, c.file_language) for c in self._chunk_buffer)
        self._chunk_buffer.clear()

    def delete_chunks_for_blobs(self, blob_shas: set[str], file_language: str) -> None:
        """Delete the given blobs' chunks extracted as *file_language*.

        Called before re-inserting chunks when a blob must be
        re-extracted. Chunks are content-addressed and shared across
        repos and paths, so a blob's chunks are deleted globally;
        chunking is deterministic per extraction serial, so every repo
        sees the same chunks for the blob. The delete is confined to one
        `file_language`: the same bytes extracted as another language are
        different chunks, and taking them too would leave those files
        with nothing to find.
        """
        if not blob_shas:
            return
        self._require_active()
        self._cursor.execute(
            _DELETE_CHUNKS_FOR_BLOBS_SQL,
            {"blob_shas": list(blob_shas), "file_language": file_language},
        )
        self._chunks_modified = True

    def add_watched_refs(self, refs: list[str]) -> None:
        """Add *refs* to the repo's watch set (idempotent on the PK)."""
        if not refs:
            return
        self._require_active()
        self._cursor.execute(_ADD_WATCHED_REFS_SQL, {"repo_id": self._repo.repo_id, "refs": refs})

    def remove_watched_refs(self, refs: list[str]) -> None:
        """Remove *refs* from the repo's watch set."""
        if not refs:
            return
        self._require_active()
        self._cursor.execute(
            _REMOVE_WATCHED_REFS_SQL, {"repo_id": self._repo.repo_id, "refs": refs}
        )

    def insert_snapshots(self, snapshots: list[FileSnapshot]) -> None:
        """Batch insert snapshots."""
        if not snapshots:
            return
        self._bulk_insert(
            _UPSERT_SNAPSHOTS_SQL, staged_file_snapshots(snapshots, self._repo.repo_id)
        )

    def replace_snapshots(self, snapshots: list[FileSnapshot], *, snapshot_sha: str) -> None:
        """Atomically replace all of the repo's snapshots at *snapshot_sha*."""
        self._flush_chunks()
        self._require_active()
        self._cursor.execute(
            _DELETE_SNAPSHOTS_SQL, {"repo_id": self._repo.repo_id, "snapshot_sha": snapshot_sha}
        )
        self.insert_snapshots(snapshots)

    def insert_edges(self, edges: list[Edge], *, snapshot_sha: str) -> None:
        """Batch insert edges of the repo's snapshot *snapshot_sha*."""
        if not edges:
            return
        self._bulk_insert(_INSERT_EDGES_SQL, staged_edges(edges, at=self._repo.at(snapshot_sha)))

    def replace_edges(self, edges: list[Edge], *, snapshot_sha: str) -> None:
        """Atomically replace all edges of the repo's snapshot *snapshot_sha*."""
        self._require_active()
        self._cursor.execute(
            _DELETE_EDGES_SQL, {"repo_id": self._repo.repo_id, "snapshot_sha": snapshot_sha}
        )
        self.insert_edges(edges, snapshot_sha=snapshot_sha)

    def update_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        truncated: list[bool] | None = None,
    ) -> None:
        """Batch-update embedding vectors via a polars staging frame.

        Updates by chunk `id` globally — the embedding lives on the
        shared content-addressed chunk row.
        """
        if not ids:
            return
        self._require_active()
        self._flush_chunks()
        if truncated is None:
            truncated = [False] * len(ids)
        frame = staged_embeddings(ids, embeddings, truncated)
        with registered_views(self._cursor, _emb_stg=frame) as cur:
            cur.execute(_UPDATE_EMBEDDINGS_SQL)

    def mark_indexed(self, *, snapshot_sha: str) -> None:
        """Record the repo's snapshot *snapshot_sha* as fully indexed."""
        self._require_active()
        self._cursor.execute(
            _MARK_INDEXED_SQL, {"repo_id": self._repo.repo_id, "snapshot_sha": snapshot_sha}
        )

    def drop_snapshot(self, *, snapshot_sha: str) -> GcCounts:
        """Remove all trace of the repo's snapshot *snapshot_sha*."""
        self._require_active()
        params = {"repo_id": self._repo.repo_id, "snapshot_sha": snapshot_sha}
        commit_rows = self._cursor.execute(_DROP_SNAPSHOT_SQL, params).fetchall()
        snap_rows = self._cursor.execute(_DELETE_SNAPSHOTS_SQL, params).fetchall()
        edge_rows = self._cursor.execute(_DELETE_EDGES_SQL, params).fetchall()
        chunk_rows = self._cursor.execute(_SWEEP_ORPHAN_CHUNKS_SQL).fetchall()
        chunks_deleted = int(chunk_rows[0][0])
        if chunks_deleted > 0:
            self._chunks_modified = True
        return GcCounts(
            snapshots=int(commit_rows[0][0]),
            file_snapshots=int(snap_rows[0][0]),
            edges=int(edge_rows[0][0]),
            chunks=chunks_deleted,
        )

    def cleanup(self) -> GcCounts:
        """Remove the session's repo's residue and what it leaves unclaimed.

        1. Delete file snapshots and edges of the repo's snapshots never
           marked indexed: what a build that stopped part-way left.
        2. Delete chunks no file snapshot in the project references.
        3. Delete the repo's edges whose snapshot has no file snapshots.

        A build runs it once its snapshot is marked indexed.  Returns
        counts of rows removed.
        """
        self._require_active()
        snap_rows = self._cursor.execute(
            _SWEEP_ORPHAN_SNAPSHOTS_SQL, {"repo_id": self._repo.repo_id}
        ).fetchall()
        orphan_edge_rows = self._cursor.execute(
            _SWEEP_ORPHAN_EDGES_SQL, {"repo_id": self._repo.repo_id}
        ).fetchall()
        chunk_rows = self._cursor.execute(_PRUNE_CHUNKS_SQL).fetchall()
        stale_edge_rows = self._cursor.execute(
            _PRUNE_EDGES_SQL, {"repo_id": self._repo.repo_id}
        ).fetchall()
        chunks_deleted = int(chunk_rows[0][0])
        if chunks_deleted > 0:
            self._chunks_modified = True
        return GcCounts(
            file_snapshots=int(snap_rows[0][0]),
            edges=int(orphan_edge_rows[0][0]) + int(stale_edge_rows[0][0]),
            chunks=chunks_deleted,
        )
