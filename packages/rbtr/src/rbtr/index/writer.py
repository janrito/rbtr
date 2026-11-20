"""The index's write session: every change to a project's data goes through `WriteSession`."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType

import dataframely as dy
import duckdb
import polars as pl
import structlog

from rbtr.config import config
from rbtr.domain.edge_frames import EdgeChunkRow, InferredEdgeRow
from rbtr.domain.models import FileSnapshot, GcCounts, Repo, SnapshotRef
from rbtr.index import load_sql
from rbtr.index.db import build_fts, registered_views
from rbtr.index.staging import (
    ChunkStagingRow,
    EmbeddingStagingRow,
    SerialMapRow,
    staged_edges,
    staged_file_snapshots,
)

CURRENT_BLOBS_SQL = load_sql("current_blobs.sql")
_COUNT_UNCLAIMED_BLOBS_SQL = load_sql("count_unclaimed_blobs.sql")
SNAPSHOT_LANGUAGES_SQL = load_sql("snapshot_languages.sql")
_DELETE_CHUNKS_FOR_BLOBS_SQL = load_sql("delete_chunks_for_blobs.sql")
_ADD_WATCHED_REFS_SQL = load_sql("insert_watched_refs.sql")
_REMOVE_WATCHED_REFS_SQL = load_sql("delete_watched_refs.sql")
_DELETE_EDGES_SQL = load_sql("delete_edges.sql")
_DELETE_SNAPSHOTS_SQL = load_sql("delete_snapshots.sql")
_DROP_SNAPSHOT_SQL = load_sql("drop_snapshot.sql")
_EDGE_CHUNKS_SQL = load_sql("edge_chunks.sql")
_INSERT_EDGES_SQL = load_sql("insert_edges.sql")
_MARK_INDEXED_SQL = load_sql("mark_indexed.sql")
_PRUNE_CHUNKS_SQL = load_sql("prune_chunks.sql")
_PRUNE_EDGES_SQL = load_sql("prune_edges.sql")
_SWEEP_ORPHAN_CHUNKS_SQL = load_sql("sweep_orphan_chunks.sql")
_UPDATE_EMBEDDINGS_SQL = load_sql("update_embeddings.sql")
_UPSERT_CHUNKS_SQL = load_sql("upsert_chunks.sql")
_UPSERT_SNAPSHOTS_SQL = load_sql("upsert_snapshots.sql")
SNAPSHOTS_LISTING_BLOBS_SQL = load_sql("snapshots_listing_blobs.sql")
SNAPSHOT_FILES_SQL = load_sql("snapshot_files.sql")

log = structlog.get_logger(__name__)


class WriteSession:
    """One transaction on one project's file, writing for one of its repos.

    Use as a context manager via `Index.session(repo)`:

    ```python
    with index.session(repo) as ws:
        ws.add_chunks(staged)                      # buffered, content-addressed
        ws.replace_edges(edges, at=repo.at(sha))   # immediate
    # exit: flush buffer + commit + FTS rebuild; rollback on exception
    ```

    Every row a write adds belongs to the session's repo, so no method
    takes one, except edges: re-extracting a blob changes the chunks of
    every snapshot that lists it, in any of the project's repos, so edges
    are read and replaced at a snapshot (`at`), which may be another
    checkout's.  Chunks are the project's: its checkouts share them.  The
    reads the transaction depends on (`current_blobs`,
    `snapshot_languages`) run on the session's cursor, so they see
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
        self._chunk_buffer: list[dy.DataFrame[ChunkStagingRow]] = []
        self._buffered = 0

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
        """Execute *sql* against *frame*, bound as the view `_stg`."""
        self._require_active()
        with registered_views(self._cursor, _stg=frame) as cur:
            cur.execute(sql)

    def current_blobs(
        self, blob_shas: Sequence[str], serials: dy.DataFrame[SerialMapRow]
    ) -> set[tuple[str, str]]:
        """The current `(blob_sha, file_language)` extractions of *blob_shas*.

        Each returned pair has stored chunks and every chunk's language
        and extraction serial matches *serials*, including embedded
        languages. Results span the session's project and reflect
        transaction-local deletions. An absent blob has no result.
        """
        if not blob_shas:
            return set()
        self._require_active()
        with registered_views(self._cursor, _serial_map=serials) as cur:
            rows = cur.sql(CURRENT_BLOBS_SQL, params={"blob_shas": list(blob_shas)}).fetchall()
            return {(blob_sha, language) for blob_sha, language in rows}

    def snapshot_languages(self, file_paths: Sequence[str]) -> dict[str, str]:
        """Each of *file_paths*' language, as one of the repo's snapshots recorded it.

        A path no snapshot of the repo holds is absent.
        """
        if not file_paths:
            return {}
        self._require_active()
        rows = self._cursor.sql(
            SNAPSHOT_LANGUAGES_SQL,
            params={"repo_id": self._repo.repo_id, "file_paths": list(file_paths)},
        ).fetchall()
        return dict(rows)

    def add_chunks(self, chunks: dy.DataFrame[ChunkStagingRow]) -> None:
        """Buffer content-addressed chunks for insertion.

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
        self._chunk_buffer.append(chunks)
        self._buffered += chunks.height
        if self._buffered >= config.insert_batch_size:
            self._flush_chunks()

    def _flush_chunks(self) -> None:
        """Write buffered chunks to DuckDB in one batch."""
        if not self._chunk_buffer:
            return
        chunks = pl.concat(self._chunk_buffer)
        self._bulk_insert(_UPSERT_CHUNKS_SQL, chunks)
        self._chunks_modified = True
        self._inserted_blobs.update(chunks.select("blob_sha", "file_language").unique().rows())
        self._chunk_buffer.clear()
        self._buffered = 0

    def replace_chunks(self, chunks: dy.DataFrame[ChunkStagingRow]) -> None:
        """Buffer *chunks* in place of what is stored for their blobs, as their file languages.

        A blob extracted again may give chunks with other ids than before
        (its language may have changed), so the chunks stored for each
        blob and file language in *chunks* that *chunks* does not hold are
        deleted. A chunk it holds again keeps its embedding while its
        name and content are unchanged (`upsert_chunks.sql`), which is
        what the model reads.
        """
        blobs = chunks.group_by("file_language").agg(pl.col("blob_sha").unique(), pl.col("id"))
        for file_language, blob_shas, ids in blobs.iter_rows():
            self.delete_chunks_for_blobs(set(blob_shas), file_language, keep=set(ids))
        self.add_chunks(chunks)

    def delete_chunks_for_blobs(
        self, blob_shas: set[str], file_language: str, *, keep: set[str] | None = None
    ) -> None:
        """Delete the given blobs' chunks extracted as *file_language*, except the ids in *keep*.

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
            {
                "blob_shas": list(blob_shas),
                "file_language": file_language,
                "keep": sorted(keep or ()),
            },
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

    def replace_snapshots(self, snapshots: list[FileSnapshot], *, snapshot_sha: str) -> None:
        """Atomically replace all of the repo's snapshots at *snapshot_sha*."""
        self._flush_chunks()
        self._require_active()
        self._cursor.execute(
            _DELETE_SNAPSHOTS_SQL, {"repo_id": self._repo.repo_id, "snapshot_sha": snapshot_sha}
        )
        if snapshots:
            self._bulk_insert(
                _UPSERT_SNAPSHOTS_SQL, staged_file_snapshots(snapshots, self._repo.repo_id)
            )

    def snapshots_listing(
        self, blobs: set[tuple[str, str]], *, repos: Sequence[Repo]
    ) -> list[SnapshotRef]:
        """The indexed snapshots of *repos* that list one of *blobs*, each a `(blob_sha, file_language)` extraction.

        Any repo of the project, the session's included: these are the
        snapshots whose chunks change when the blobs are extracted again.
        """
        if not blobs:
            return []
        self._require_active()
        pairs = pl.DataFrame(sorted(blobs), schema=["blob_sha", "file_language"], orient="row")
        with registered_views(self._cursor, _stg=pairs) as cur:
            rows = cur.sql(SNAPSHOTS_LISTING_BLOBS_SQL).fetchall()
        by_id = {repo.repo_id: repo for repo in repos}
        return [by_id[repo_id].at(sha) for repo_id, sha in rows if repo_id in by_id]

    def snapshot_files(self, *, at: SnapshotRef) -> dict[str, str]:
        """Each file of the snapshot *at*, and the blob it holds."""
        self._require_active()
        rows = self._cursor.sql(
            SNAPSHOT_FILES_SQL,
            params={"repo_id": at.repo.repo_id, "snapshot_sha": at.snapshot_sha},
        ).fetchall()
        return dict(rows)

    def edge_chunks(self, *, at: SnapshotRef) -> dy.DataFrame[EdgeChunkRow]:
        """Every located chunk of the snapshot *at*, as edge inference reads it.

        Ordered by path and position, so a duplicated chunk appears once per
        location and the first chunk of a file comes first.
        """
        self._require_active()
        self._flush_chunks()
        return (
            self._cursor.sql(
                _EDGE_CHUNKS_SQL,
                params={"repo_id": at.repo.repo_id, "snapshot_sha": at.snapshot_sha},
            )
            .pl()
            .pipe(EdgeChunkRow.validate, cast=True)
        )

    def replace_edges(self, edges: dy.DataFrame[InferredEdgeRow], *, at: SnapshotRef) -> None:
        """Atomically replace all edges of the snapshot *at*."""
        self._require_active()
        self._cursor.execute(
            _DELETE_EDGES_SQL, {"repo_id": at.repo.repo_id, "snapshot_sha": at.snapshot_sha}
        )
        if not edges.is_empty():
            self._bulk_insert(_INSERT_EDGES_SQL, staged_edges(edges, at=at))

    def update_embeddings(self, embeddings: dy.DataFrame[EmbeddingStagingRow]) -> None:
        """Set each chunk's embedding and truncation flag from *embeddings*.

        Updates by chunk `id` globally — the embedding lives on the
        shared content-addressed chunk row.
        """
        if embeddings.is_empty():
            return
        self._require_active()
        self._flush_chunks()
        with registered_views(self._cursor, _emb_stg=embeddings) as cur:
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
        """Remove what no snapshot claims: rows a snapshot's rebuild or drop leaves.

        1. Delete chunks no file snapshot in the project references.
        2. Delete the repo's edges whose snapshot has no file snapshots.

        A build runs it once its snapshot is marked indexed.  Returns
        counts of rows removed.
        """
        self._require_active()
        chunk_rows = self._cursor.execute(_PRUNE_CHUNKS_SQL).fetchall()
        stale_edge_rows = self._cursor.execute(
            _PRUNE_EDGES_SQL, {"repo_id": self._repo.repo_id}
        ).fetchall()
        chunks_deleted = int(chunk_rows[0][0])
        if chunks_deleted > 0:
            self._chunks_modified = True
        return GcCounts(edges=int(stale_edge_rows[0][0]), chunks=chunks_deleted)
