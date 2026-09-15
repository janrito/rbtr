"""DuckDB storage — schema, reads, and search for the code index.

The store manages four tables:

- `file_snapshots` maps a commit SHA to its file tree (path → blob SHA).
- `chunks` holds indexed content, keyed by blob SHA so identical files
  across commits are stored once.
- `edges` records relationships between chunks, scoped per commit.
- `indexed_snapshots` tracks which `(repo, commit)` pairs have been
  fully indexed.

All commit-scoped queries join through `file_snapshots` to resolve
which chunks belong to a given snapshot.

## Type architecture

`IndexStore` owns the DuckDB connection, thread-local cursor
cache, FTS index, and all read methods.  Write operations
live on `WriteSession` (in `writer.py`), obtained via
`IndexStore.session()`.  The session is a context manager
that wraps all writes in a transaction: commit + FTS rebuild
on clean exit, rollback on exception.

The store defaults to read-only (`writable=False`).  Pass
`writable=True` to enable `session()`.  This prevents
accidental writes from search handlers.

## FTS rebuild contract

DuckDB FTS does not auto-update after INSERT or DELETE.
The FTS index is rebuilt by `WriteSession` after commit
when chunks were modified.  On open, the persisted index
(if any) is queryable immediately; otherwise the first
session that inserts chunks rebuilds it.

## Blob dedup and language-change invalidation

See the "Blob dedup and language invalidation" section in
`ARCHITECTURE.md` for the full flow.  `blob_is_current` gates
extraction by `(blob_sha, file_language)` — the language the *file* was
read as, not a chunk's own, so an empty `.html` is still extracted after
an identical empty `.py`.  The `blob_is_current`
docstring documents the semantics and prose special case.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import dataframely as dy
import duckdb
import polars as pl

# duckdb requires this imported before any thread uses a connection.
import pyarrow  # type: ignore[import-untyped]  # noqa: F401
import structlog

from rbtr.config import WeightTriple, config
from rbtr.domain.models import (
    Chunk,
    ChunkKind,
    EdgeKind,
    QueryKind,
    Repo,
    ScoredChunk,
    SnapshotCounts,
    SnapshotRange,
    SnapshotRef,
)
from rbtr.domain.tokenise import tokenise_code
from rbtr.errors import IndexLockedError, IndexNotBuiltError, IndexSchemaTooNewError, RbtrError
from rbtr.git import worktree_tree_sha
from rbtr.index import load_sql
from rbtr.index.constants import SCHEMA_VERSION
from rbtr.index.embeddings import Embedder
from rbtr.index.reranker import Reranker
from rbtr.index.results import (
    ChangedSymbolRow,
    ChunkContentRow,
    ChunkPathResultRow,
    ChunkResultRow,
    EdgeResultRow,
    InboundDegreeResultRow,
    InboundRefResultRow,
    ScoredChunkResultRow,
    SnapshotCountsRow,
    _decode_metadata,
    chunk_ids_view,
    file_paths_view,
    frame_to_chunks,
    frame_to_snapshot_counts,
    serial_map_view,
    snapshot_refs_view,
)
from rbtr.index.search import search
from rbtr.index.writer import WriteSession

log = structlog.get_logger(__name__)


# Pre-load all SQL at import time so file I/O is not on the hot path.
_GET_CHUNKS_SQL = load_sql("get_chunks.sql")
_GET_EDGES_SQL = load_sql("get_edges.sql")
_INBOUND_REFS_SQL = load_sql("inbound_refs.sql")
_CHANGED_SYMBOLS_SQL = load_sql("changed_symbols.sql")
_SEARCH_BY_NAME_SQL = load_sql("search_by_name.sql")
_SEARCH_SIMILAR_SQL = load_sql("search_similar.sql")
_SEARCH_FULLTEXT_SQL = load_sql("search_fulltext.sql")
_COUNT_ORPHAN_CHUNKS_SQL = load_sql("count_orphan_chunks.sql")
_COUNT_GC_CHUNK_SPLIT_SQL = load_sql("count_gc_chunk_split.sql")
_INBOUND_DEGREE_SQL = load_sql("inbound_degree.sql")
_BLOB_IS_CURRENT_SQL = load_sql("blob_is_current.sql")
_GET_SCHEMA_VERSION_SQL = load_sql("get_schema_version.sql")
_GET_REPO_SQL = load_sql("get_repo.sql")
_LIST_REPOS_SQL = load_sql("list_repos.sql")
_GET_CHUNK_PATHS_SQL = load_sql("get_chunk_paths.sql")
_CHUNK_COUNTS_BY_SNAPSHOT_SQL = load_sql("chunk_counts_by_snapshot.sql")
_DISTINCT_CHUNK_LANGUAGES_SQL = load_sql("distinct_chunk_languages.sql")
_HAS_INDEXED_SQL = load_sql("has_indexed.sql")
_LIST_INDEXED_COMMITS_SQL = load_sql("list_indexed_snapshots.sql")
_LIST_WATCHED_REFS_SQL = load_sql("list_watched_refs.sql")
_COUNT_FILE_SNAPSHOTS_SQL = load_sql("count_file_snapshots.sql")
_COUNT_EDGES_SQL = load_sql("count_edges.sql")
_GET_SNAPSHOT_LANGUAGE_SQL = load_sql("get_snapshot_language.sql")
_UNEMBEDDED_CHUNK_IDS_SQL = load_sql("unembedded_chunk_ids.sql")
_GET_CHUNKS_BY_ID_SQL = load_sql("get_chunks_by_id.sql")
_HAS_FTS_INDEX_SQL = load_sql("has_fts_index.sql")
_DROP_FTS_INDEX_SQL = load_sql("drop_fts_index.sql")
_WIPE_SCHEMA_SQL = load_sql("wipe_schema.sql")
_DATA_SIZE_BYTES_SQL = load_sql("data_size_bytes.sql")


# ── Row mapping ──────────────────────────────────────────────────────


def _parse_calver(version: str) -> tuple[int, ...]:
    """Parse a calver version to an int tuple for ordering.

    Compared as ints, not lexically (`2026.5.6 < 2026.5.10`).  An
    unparseable version sorts oldest (`()`) so a foreign or corrupt
    stored version rebuilds rather than being mistaken for newer.
    """
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return ()


# ── IndexStore ─────────────────────────────────────────────────────────


class IndexStore:
    """DuckDB-backed storage for the code index.

    Owns the DuckDB connection, thread-local cursor cache, FTS
    index, and read methods.  Write operations live on
    `WriteSession`, obtained via `session()`.

    DuckDB connections are **not** thread-safe, but cursors
    obtained via `connection.cursor()` are isolated per-call.
    The `_cursor` property caches one cursor per thread so the
    build thread and search handlers can share one instance.
    """

    def __init__(self, db_path: Path | str | None = None, *, writable: bool = False) -> None:
        self.db_path: str | None = str(db_path) if db_path else None
        self._writable = writable
        self._bootstrapped = False
        self._repo_cache: dict[str, int] = {}
        if db_path is not None:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        dsn = str(db_path) if db_path else ":memory:"
        try:
            self._con = duckdb.connect(dsn)
        except duckdb.IOException as exc:
            msg = str(exc)
            if "lock" in msg.lower():
                log.exception("duckdb_lock_conflict")
                locked_msg = (
                    "Index database is locked by another process. "
                    "If the daemon is running, route commands through it "
                    "(`rbtr daemon status` to check)."
                )
                raise IndexLockedError(locked_msg) from exc
            raise
        self._load_fts(self._con)
        self._local = threading.local()
        # Enforce the schema version *while holding the connection* (and
        # thus DuckDB's exclusive lock), so a wipe is done in place and
        # never unlinks the file -- unlinking would let a second process
        # open a fresh DB at the same path and defeat the lock.
        if db_path is not None:
            self._enforce_schema_version()
        if writable:
            with self.session():
                pass  # schema DDL + embedding version check

    def _enforce_schema_version(self) -> None:
        """Keep, wipe, or refuse the open DB by its stored schema version.

        Runs under the held connection.  An equal version keeps the
        DB; a newer running binary wipes and rebuilds it in place; an
        *older* running binary refuses (raises `IndexSchemaTooNewError`)
        rather than destroy an index a newer rbtr wrote.
        """
        try:
            rows = self._con.execute(_GET_SCHEMA_VERSION_SQL).fetchall()
        except duckdb.CatalogException:
            return  # no meta table: fresh DB, the writable bootstrap builds it
        stored = str(rows[0][0]) if rows else ""
        if stored == SCHEMA_VERSION:
            return
        if _parse_calver(stored) > _parse_calver(SCHEMA_VERSION):
            self._con.close()
            raise IndexSchemaTooNewError(stored=stored, code=SCHEMA_VERSION)
        if not self._writable:
            self._con.close()
            msg = (
                f"Index schema {stored or 'unknown'} predates this rbtr "
                f"({SCHEMA_VERSION}); run `rbtr index` to rebuild it."
            )
            raise RbtrError(msg)
        log.warning("index_schema_changed", stored=stored or "none", current=SCHEMA_VERSION)
        if self._con.execute(_HAS_FTS_INDEX_SQL).fetchone():
            self._con.execute(_DROP_FTS_INDEX_SQL)
        self._con.execute(_WIPE_SCHEMA_SQL)

    @classmethod
    def from_config(cls, *, writable: bool = False) -> IndexStore:
        """Open the database at the central DB path."""
        return cls(config.db_path, writable=writable)

    @property
    def _cursor(self) -> duckdb.DuckDBPyConnection:
        """Thread-local cursor — safe to access from any thread.

        Caches one cursor per thread to avoid the overhead of
        `connection.cursor()` on every operation (which triggers
        `getcwd` + `stat` syscalls in DuckDB).
        """
        cur = getattr(self._local, "cur", None)
        if cur is None or self._local.built_against is not self._con:
            # First use on this thread, or compaction published a new
            # connection: bind a fresh cursor to the current one. A read
            # already in progress keeps the cursor it captured (see
            # `reader`), so a swap never tears it.
            cur = self._con.cursor()
            self._local.cur = cur
            self._local.built_against = self._con
        return cur

    @contextmanager
    def reader(self, **frames: pl.DataFrame) -> Iterator[duckdb.DuckDBPyConnection]:
        """A cursor for the block, with *frames* registered as temp views.

        The query a caller writes decides what it reads; *frames* bind
        the values it joins against, each under its keyword name, and
        are dropped when the block ends.

        Capturing the cursor once keeps a multi-statement read atomic
        against a concurrent compaction swap: the register, the query,
        and the unregister all run on the same connection, which stays
        alive for this read. It also replaces the register/try/finally/
        unregister boilerplate each such read would otherwise repeat.
        """
        cur = self._cursor
        for name, frame in frames.items():
            cur.register(name, frame)
        try:
            yield cur
        finally:
            for name in frames:
                cur.unregister(name)

    def session(self) -> WriteSession:
        """Create a `WriteSession` for use as a context manager.

        **Concurrency warning:** in the daemon, write sessions must
        only be opened from the job worker thread (serialised by
        `_write_sem`).  The watcher, RPC handlers, and embed
        preemption checks must be read-only.

        Raises `RuntimeError` if the store was opened read-only.
        """
        if not self._writable:
            msg = "session() requires a writable IndexStore (pass writable=True)"
            raise RuntimeError(msg)
        return WriteSession(self)

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    def data_size_bytes(self) -> int:
        """Logical size allocated inside the main database file, in bytes.

        `block_size * total_blocks` from `pragma_database_size` — the
        in-file pages including free space kept for reuse, so it drops
        only when the file is rewritten (compacted), never from a plain
        delete. This is the measure of compaction reclaim. It excludes
        the write-ahead log and reads zero until the WAL is consolidated,
        so it is not the on-disk footprint (see `disk_size_bytes`).
        """
        row = self._cursor.execute(_DATA_SIZE_BYTES_SQL).fetchone()
        return int(row[0]) if row is not None else 0

    def disk_size_bytes(self) -> int:
        """Total bytes the index occupies on disk (database file + WAL).

        The honest footprint a user sees: it counts the actual files, so
        it is non-zero as soon as anything is committed and never depends
        on whether the write-ahead log has been consolidated into the
        main file. Zero for an in-memory store.
        """
        if self.db_path is None:
            return 0
        db = Path(self.db_path)
        total = db.stat().st_size if db.exists() else 0
        wal = db.with_name(f"{db.name}.wal")
        if wal.exists():
            total += wal.stat().st_size
        return total

    def _load_fts(self, con: duckdb.DuckDBPyConnection) -> None:
        """Install and load the full-text-search extension on *con*."""
        con.install_extension("fts")
        con.load_extension("fts")

    def _adopt_connection(self, con: duckdb.DuckDBPyConnection) -> None:
        """Publish a freshly-opened connection as the live one, lock-free.

        Compaction opens the rewritten file as its own connection
        (`WriteSession.compact`) and hands it here. Assigning `self._con`
        is atomic under the GIL, so a reader sees either the old
        connection or the new one, never a torn state; each thread's
        cached cursor was built against the old connection, so `_cursor`
        rebinds it on next use. The old connection is **not** closed here
        — it stays alive for readers still using it and closes itself by
        refcount once the last has rebound, so no read is ever cut off
        mid-query.
        """
        self._load_fts(con)
        self._con = con

    def get_repo_id(self, path: str) -> int | None:
        """Return the repo_id for *path*, or None if not registered.

        *path* is matched exactly, as stored at registration — it is not
        normalised, so pass a canonical path.  A repo whose checkout has
        since been deleted is still found, which is how it gets forgotten.
        """
        row = self._cursor.execute(_GET_REPO_SQL, {"path": path}).fetchone()
        return int(row[0]) if row else None

    def resolve_repo(self, repo: str) -> int:
        """Return the repo_id for *repo*, raising if not registered.

        As with `get_repo_id`, *repo* is matched exactly and must already
        be canonical.  Results are cached for the lifetime of the store.
        """
        cached = self._repo_cache.get(repo)
        if cached is not None:
            return cached
        repo_id = self.get_repo_id(repo)
        if repo_id is None:
            msg = f"Repo not registered: {repo}"
            raise RbtrError(msg)
        self._repo_cache[repo] = repo_id
        return repo_id

    def list_repos(self) -> list[Repo]:
        """Return all registered repos."""
        rows = self._cursor.execute(_LIST_REPOS_SQL).fetchall()
        return [Repo(repo_id=int(r[0]), repo_path=str(r[1])) for r in rows]

    def list_watched_refs(self, repo_id: int) -> list[str]:
        """Return the repo's watched refs (symbolic names), sorted by name."""
        rows = self._cursor.execute(_LIST_WATCHED_REFS_SQL, {"repo_id": repo_id}).fetchall()
        return [str(r[0]) for r in rows]

    def indexed_worktree_ref(self, repo_path: str, repo_id: int) -> SnapshotRef | None:
        """The worktree's tree SHA, when the tree is dirty and indexed.

        `None` when the worktree matches HEAD's tree, when the repo
        is gone, or when that tree has never been indexed — in each
        case the caller wants a commit instead.
        """
        tree_sha = worktree_tree_sha(repo_path)
        if tree_sha is None:
            return None
        at = SnapshotRef(repo_id=repo_id, snapshot_sha=tree_sha)
        return at if self.has_indexed(at=at) else None

    def latest_indexed_ref(self, repo_id: int) -> SnapshotRef | None:
        """The most recently indexed snapshot for one repo, if any.

        `None` when the repo has never been indexed.
        """
        indexed = self.list_indexed_snapshots(repo_id)
        if not indexed:
            return None
        return SnapshotRef(repo_id=repo_id, snapshot_sha=indexed[0][0])

    def latest_ref(self, repo: Repo) -> SnapshotRef | None:
        """Resolve the most recent indexed ref for one repo.

        Prefers the current worktree tree SHA when the worktree is
        dirty and that tree has been indexed; otherwise falls back
        to the newest indexed commit.  Returns `None` when the repo
        has no indexed commits at all.
        """
        dirty = self.indexed_worktree_ref(repo.repo_path, repo.repo_id)
        return dirty if dirty is not None else self.latest_indexed_ref(repo.repo_id)

    def list_latest_refs(self) -> list[SnapshotRef]:
        """Return one `SnapshotRef` per registered repo with indexed data.

        Applies `latest_ref` to every repo from `list_repos`,
        skipping repos that have never been indexed.  The basis for
        cross-repo search: the returned list is passed straight to
        `search()`.
        """
        refs: list[SnapshotRef] = []
        for repo in self.list_repos():
            ref = self.latest_ref(repo)
            if ref is not None:
                refs.append(ref)
        return refs

    # ── Completion tracking (indexed_snapshots) ──────────────────────

    def has_indexed(self, *, at: SnapshotRef) -> bool:
        """Return whether the snapshot at *at* was fully indexed."""
        row = self._cursor.execute(
            _HAS_INDEXED_SQL, {"repo_id": at.repo_id, "snapshot_sha": at.snapshot_sha}
        ).fetchone()
        return row is not None

    def list_indexed_snapshots(self, repo_id: int) -> list[tuple[str, str]]:
        """Return `(snapshot_sha, indexed_at)` for this repo, newest first."""
        rows = self._cursor.execute(_LIST_INDEXED_COMMITS_SQL, {"repo_id": repo_id}).fetchall()
        return [
            (
                str(r[0]),
                str(r[1]),
            )
            for r in rows
        ]

    def count_file_snapshots(self, *, at: SnapshotRef) -> int:
        """Return the number of `file_snapshots` rows for this commit.

        Read-only. Used by dry-run GC reporting.
        """
        row = self._cursor.execute(
            _COUNT_FILE_SNAPSHOTS_SQL, {"repo_id": at.repo_id, "snapshot_sha": at.snapshot_sha}
        ).fetchone()
        return int(row[0]) if row else 0

    def count_edges(self, *, at: SnapshotRef) -> int:
        """Return the number of `edges` rows for this commit.

        Read-only. Used by dry-run GC reporting.
        """
        row = self._cursor.execute(
            _COUNT_EDGES_SQL,
            {"repo_id": at.repo_id, "snapshot_sha": at.snapshot_sha},
        ).fetchone()
        return int(row[0]) if row else 0

    def count_orphan_chunks(self) -> int:
        """Count chunks not referenced by any file snapshot in any repo.

        The chunk store is content-addressed and shared, so orphan
        status is global: a chunk is an orphan iff no snapshot
        anywhere references its `(blob_sha, file_language)`.
        """
        row = self._cursor.execute(_COUNT_ORPHAN_CHUNKS_SQL).fetchone()
        return int(row[0]) if row else 0

    def count_gc_chunk_split(self, repo_id: int, drop_shas: list[str]) -> tuple[int, int]:
        """Split a GC drop set's chunks into `(dropped, kept_shared)`.

        *dropped* is the chunks the drop set would free; *kept_shared*
        is candidate chunks retained because a ref outside the drop set
        (another ref of this repo, or any other repo) still references
        their `(blob_sha, file_language)`.  Read-only — it computes the
        split from the reference graph without simulating the drop, so
        it serves dry-run and real runs alike.
        """
        if not drop_shas:
            return (0, 0)
        row = self._cursor.execute(
            _COUNT_GC_CHUNK_SPLIT_SQL, {"repo_id": repo_id, "drop_shas": drop_shas}
        ).fetchone()
        if row is None:
            return (0, 0)
        return (int(row[0]), int(row[1]))

    # ── Reads ────────────────────────────────────────────────────────

    def blob_is_current(self, blob_sha: str, language: str, serials: dict[str, int]) -> bool:
        """Check whether *blob_sha* is up to date for host *language*.

        This is the blob-dedup gate: the orchestrator calls it
        before extracting a file.  If True, the blob's chunks are
        current and valid — skip.  If False, the file needs
        (re-)extraction.  The check is **global** (the chunk store
        is content-addressed and shared), so a blob already parsed
        by any repo is reused — no re-parse when a second
        repo/worktree indexes it.

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
        with self.reader(_serial_map=serial_map_view(serials)) as cur:
            row = cur.execute(
                _BLOB_IS_CURRENT_SQL, {"blob_sha": blob_sha, "language": language}
            ).fetchone()
        return bool(row[0]) if row and row[0] is not None else False

    def get_snapshot_language(self, file_path: str, *, repo_id: int) -> str:
        """Return the detected language from any existing snapshot.

        Returns `''` if the file has never been indexed or was
        indexed as plaintext.
        """
        row = self._cursor.execute(
            _GET_SNAPSHOT_LANGUAGE_SQL,
            {"repo_id": repo_id, "file_path": file_path},
        ).fetchone()
        return str(row[0]) if row else ""

    def chunk_counts_by_snapshot(
        self, *, repo_id: int | None = None
    ) -> list[tuple[SnapshotRef, SnapshotCounts]]:
        """Return chunk and embedded counts per indexed snapshot, one pair each.

        Spans every repo unless *repo_id* narrows it.  One grouped pass
        whatever the scope, so the cost does not grow with the number of
        snapshots asked about — which matters because reading
        `chunks.embedding` is most of the work.

        Every snapshot in `indexed_snapshots` gets a pair; one holding no
        chunks counts zero.  Ordered most recently indexed first, ties
        broken by `snapshot_sha`.
        """
        return frame_to_snapshot_counts(self._chunk_counts_frame(repo_id=repo_id))

    def _chunk_counts_frame(
        self, *, repo_id: int | None = None, snapshot_sha: str | None = None
    ) -> dy.DataFrame[SnapshotCountsRow]:
        """Back the counts readers with one validated frame.

        *snapshot_sha* narrows to a single snapshot and is meaningful
        only alongside *repo_id*, which is why it stays private: a SHA
        on its own matches that commit in every repo.
        """
        return (
            self._cursor.execute(
                _CHUNK_COUNTS_BY_SNAPSHOT_SQL,
                {"repo_id": repo_id, "snapshot_sha": snapshot_sha},
            )
            .pl()
            .pipe(SnapshotCountsRow.validate, cast=True)
        )

    def chunk_counts_for_snapshot(self, *, at: SnapshotRef) -> SnapshotCounts:
        """Return chunk and embedded counts for one indexed snapshot.

        Counts chunks, so content held at several paths counts once — the
        figure embedding progress is measured against.  A snapshot that
        holds no chunks, or was never marked indexed, reads as zero of
        zero.
        """
        frame = self._chunk_counts_frame(repo_id=at.repo_id, snapshot_sha=at.snapshot_sha)
        if frame.is_empty():
            return SnapshotCounts(total=0, embedded=0)
        row = frame.row(0, named=True)
        return SnapshotCounts(total=row["total"], embedded=row["embedded"])

    def unembedded_chunk_ids(self, *, at: SnapshotRef) -> list[str]:
        """Return the id of every chunk at *at* still lacking a vector.

        One id however many paths hold the content, because embedding
        writes the single content-addressed row they share.  The whole
        work list comes back at once: this is the one read on the embed
        path that touches `chunks.embedding`, and the caller fetches the
        chunks themselves by id.
        """
        rows = self._cursor.execute(
            _UNEMBEDDED_CHUNK_IDS_SQL,
            {"repo_id": at.repo_id, "snapshot_sha": at.snapshot_sha},
        ).fetchall()
        return [str(r[0]) for r in rows]

    def get_chunks_by_id(self, chunk_ids: list[str], *, at: SnapshotRef) -> list[Chunk]:
        """Return the named chunks as they are seen at *at*, ordered by id.

        An id that no longer resolves at *at* is left out of the
        result, which shortens the page: a chunk can be collected
        between a work list being drawn up and a page of it being read.
        """
        if not chunk_ids:
            return []
        params = {"repo_id": at.repo_id, "snapshot_sha": at.snapshot_sha}
        with self.reader(_chunk_ids=chunk_ids_view(chunk_ids)) as cur:
            frame = (
                cur.execute(_GET_CHUNKS_BY_ID_SQL, params)
                .pl()
                .pipe(_decode_metadata)
                .pipe(ChunkResultRow.validate, cast=True)
            )
        return frame_to_chunks(frame)

    def distinct_chunk_languages(self) -> set[str]:
        """Return every real language present in the store (global).

        Excludes the raw-chunk fallback pseudo-language (`''`), so the
        result is the set of languages some plugin has actually
        extracted -- a record of the plugin set that built the index.
        """
        rows = self._cursor.execute(_DISTINCT_CHUNK_LANGUAGES_SQL).fetchall()
        return {str(r[0]) for r in rows}

    def get_chunks(
        self,
        *,
        at: SnapshotRef,
        file_path: str | None = None,
        kind: ChunkKind | None = None,
        name: str | None = None,
    ) -> list[Chunk]:
        """Query chunks visible at *at* with optional filters."""
        kind_val = kind.value if kind is not None else None
        params = {
            "repo_id": at.repo_id,
            "snapshot_sha": at.snapshot_sha,
            "file_path": file_path,
            "kind": kind_val,
            "name": name,
        }
        frame = (
            self._cursor.execute(_GET_CHUNKS_SQL, params)
            .pl()
            .pipe(_decode_metadata)
            .pipe(ChunkResultRow.validate, cast=True)
        )
        return frame_to_chunks(frame)

    def inbound_refs(
        self, target_ids: list[str], *, at: SnapshotRef
    ) -> dy.DataFrame[InboundRefResultRow]:
        """Return referrers of the given target chunks at *at*.

        One row per inbound edge, resolved to the source (referrer)
        chunk's identity plus the edge kind — powers `find-refs`.
        """
        if not target_ids:
            return InboundRefResultRow.create_empty()
        with self.reader(_snapshot_refs=snapshot_refs_view([at])) as cur:
            return (
                cur.execute(_INBOUND_REFS_SQL, {"target_ids": target_ids})
                .pl()
                .pipe(InboundRefResultRow.validate, cast=True)
            )

    def chunk_contents(self, *, at: SnapshotRef) -> dy.DataFrame[ChunkContentRow]:
        """Return all chunks at *at* as a content-only frame.

        The frame is validated through `ChunkContentRow` and
        contains identity columns (`file_path`, `scope`, `name`,
        `line_start`, `line_end`, `kind`) plus `language` and
        `content`.
        """
        params = {
            "repo_id": at.repo_id,
            "snapshot_sha": at.snapshot_sha,
            "file_path": None,
            "kind": None,
            "name": None,
        }
        return (
            self._cursor.execute(_GET_CHUNKS_SQL, params)
            .pl()
            .select(
                "file_path",
                "scope",
                "name",
                "line_start",
                "line_end",
                "kind",
                "language",
                "content",
            )
            .pipe(ChunkContentRow.validate, cast=True)
        )

    def edges(
        self,
        *,
        within: list[SnapshotRef],
        source_id: str | None = None,
        target_id: str | None = None,
        kind: EdgeKind | None = None,
    ) -> dy.DataFrame[EdgeResultRow]:
        """Return edges scoped to *within* as a validated frame."""
        kind_val = kind.value if kind is not None else None
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "kind": kind_val,
        }
        with self.reader(_snapshot_refs=snapshot_refs_view(within)) as cur:
            return cur.execute(_GET_EDGES_SQL, params).pl().pipe(EdgeResultRow.validate, cast=True)

    def inbound_degrees(
        self, chunk_ids: list[str], *, within: list[SnapshotRef]
    ) -> dy.DataFrame[InboundDegreeResultRow]:
        """Return inbound edge counts for the given chunk IDs."""
        if not chunk_ids:
            return InboundDegreeResultRow.create_empty()
        with self.reader(_snapshot_refs=snapshot_refs_view(within)) as cur:
            return (
                cur.execute(_INBOUND_DEGREE_SQL, {"chunk_ids": chunk_ids})
                .pl()
                .pipe(InboundDegreeResultRow.validate, cast=True)
            )

    def changed_symbols(
        self,
        *,
        between: SnapshotRange,
        file_paths: list[str] | None = None,
    ) -> dy.DataFrame[ChangedSymbolRow]:
        """Symbol-level diff between two indexed commits.

        Returns one row per changed symbol, each labelled
        added/modified/removed in a single SQL pass. Symbol identity
        is `(file_path, name, scope)`; a head symbol is "modified"
        iff that identity exists at base but no base symbol of that
        identity has matching content (content-set membership, so a
        non-unique identity cannot fan out into spurious pairs). A
        side that is not indexed contributes no rows, so the caller
        must check both commits are present to distinguish "no
        changes" from "not indexed".

        When *file_paths* is a non-empty list, the diff is scoped to
        those files via the cursor-registered `_file_paths` semi-join
        in `changed_symbols.sql`; `None` or an empty list diffs every
        file (the `scope_all` flag bypasses the view).
        """
        params = {
            "repo_id": between.repo_id,
            "head_sha": between.head_sha,
            "base_sha": between.base_sha,
            "scope_all": not file_paths,
        }
        with self.reader(_file_paths=file_paths_view(file_paths or [])) as cur:
            return (
                cur.execute(_CHANGED_SYMBOLS_SQL, params)
                .pl()
                .pipe(_decode_metadata)
                .pipe(ChangedSymbolRow.validate, cast=True)
            )

    # ── Match (internal frame, public chunk) ─────────────────────

    def name_matches(
        self, pattern: str, *, within: list[SnapshotRef]
    ) -> dy.DataFrame[ChunkResultRow]:
        """Return name-matched chunks as a validated frame.

        Resolution is tiered: exact → case-insensitive exact →
        prefix → substring.  Only the best tier that has matches
        is returned.
        """
        with self.reader(_snapshot_refs=snapshot_refs_view(within)) as cur:
            return (
                cur.execute(
                    _SEARCH_BY_NAME_SQL,
                    {"name": pattern, "pattern": f"%{pattern}%"},
                )
                .pl()
                .pipe(_decode_metadata)
                .pipe(ChunkResultRow.validate, cast=True)
            )

    def match_by_name(self, pattern: str, *, at: SnapshotRef) -> list[Chunk]:
        """Find chunks by name with tiered resolution.

        Prefers exact matches, then case-insensitive exact, then
        prefix, then substring.  Returns only the best tier.
        """
        return frame_to_chunks(self.name_matches(pattern, within=[at]))

    def similar_matches(
        self,
        query_embeddings: list[list[float]],
        *,
        within: list[SnapshotRef],
        top_k: int = 10,
    ) -> dy.DataFrame[ScoredChunkResultRow]:
        """Return cosine-similar chunks across multiple query vectors.

        Registers a temporary polars frame of vectors, cross-joins
        it with the chunks table, and keeps the best (MAX) cosine
        similarity per chunk.  One table scan regardless of the
        number of query vectors.

        Thread-safe: `register`/`unregister` are cursor-scoped —
        the views are invisible to other cursors, so concurrent
        calls on different thread-local cursors cannot collide.
        """
        vecs_frame = pl.DataFrame({"vec": query_embeddings}).cast({"vec": pl.List(pl.Float32)})
        with self.reader(_qvecs=vecs_frame, _snapshot_refs=snapshot_refs_view(within)) as cur:
            return (
                cur.execute(_SEARCH_SIMILAR_SQL, {"top_k": top_k})
                .pl()
                .pipe(_decode_metadata)
                .pipe(ScoredChunkResultRow.validate, cast=True)
            )

    # ── FTS ──────────────────────────────────────────────────────────

    def fulltext_matches(
        self,
        query: str,
        *,
        within: list[SnapshotRef],
        top_k: int = 10,
    ) -> dy.DataFrame[ScoredChunkResultRow]:
        """Return BM25-matched chunks as a validated scored frame."""
        tokenised_query = tokenise_code(query)
        if not tokenised_query:
            return ScoredChunkResultRow.create_empty()
        with self.reader(_snapshot_refs=snapshot_refs_view(within)) as cur:
            try:
                return (
                    cur.execute(
                        _SEARCH_FULLTEXT_SQL,
                        {"tokenised_query": tokenised_query, "top_k": top_k},
                    )
                    .pl()
                    .pipe(_decode_metadata)
                    .pipe(ScoredChunkResultRow.validate, cast=True)
                )
            except duckdb.CatalogException as exc:
                raise IndexNotBuiltError from exc

    def chunk_paths(
        self, chunk_ids: list[str], *, within: list[SnapshotRef]
    ) -> dy.DataFrame[ChunkPathResultRow]:
        """Return `(id, file_path)` for the given chunk IDs."""
        if not chunk_ids:
            return ChunkPathResultRow.create_empty()
        with self.reader(_snapshot_refs=snapshot_refs_view(within)) as cur:
            return (
                cur.execute(_GET_CHUNK_PATHS_SQL, {"chunk_ids": chunk_ids})
                .pl()
                .pipe(ChunkPathResultRow.validate, cast=True)
            )

    # ── Unified search ───────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        within: list[SnapshotRef],
        top_k: int = 10,
        changed_files: set[str] | None = None,
        embedder: Embedder | None = None,
        kind: QueryKind | None = None,
        keywords: list[str] | None = None,
        variants: list[str] | None = None,
        weights: WeightTriple | None = None,
        reranker: Reranker | None = None,
        reranker_pool: int | None = None,
        reranker_blend_weight: float | None = None,
        repo_paths: dict[int, str] | None = None,
    ) -> list[ScoredChunk]:
        """Search across one or more repo refs.

        Delegates to `search.search()`.  See that function for
        details.  A one-element *within* list is a single-repo
        search; many refs fan the query across repos.  *repo_paths*
        maps `repo_id` to a path so cross-repo results carry their
        origin.
        """
        return search(
            self,
            query,
            within=within,
            top_k=top_k,
            changed_files=changed_files,
            embedder=embedder,
            kind=kind,
            keywords=keywords,
            variants=variants,
            weights=weights,
            reranker=reranker,
            reranker_pool=reranker_pool,
            reranker_blend_weight=reranker_blend_weight,
            repo_paths=repo_paths,
        )
