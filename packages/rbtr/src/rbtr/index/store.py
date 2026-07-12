"""DuckDB storage — schema, reads, and search for the code index.

The store manages four tables:

- `file_snapshots` maps a commit SHA to its file tree (path → blob SHA).
- `chunks` holds indexed content, keyed by blob SHA so identical files
  across commits are stored once.
- `edges` records relationships between chunks, scoped per commit.
- `indexed_commits` tracks which `(repo, commit)` pairs have been
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
`ARCHITECTURE.md` for the full flow.  `has_blob` gates
extraction by `(blob_sha, language)`.  The `has_blob`
docstring documents the semantics and prose special case.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import dataframely as dy
import duckdb
import polars as pl
import structlog

from rbtr.config import WeightTriple, config
from rbtr.errors import IndexNotBuiltError, RbtrError
from rbtr.git import worktree_tree_sha
from rbtr.index import load_sql
from rbtr.index.constants import SCHEMA_VERSION
from rbtr.index.embeddings import Embedder
from rbtr.index.frames import (
    ChangedSymbolRow,
    ChunkContentRow,
    ChunkPathResultRow,
    ChunkResultRow,
    EdgeResultRow,
    InboundDegreeResultRow,
    InboundRefResultRow,
    ScoredChunkResultRow,
    _decode_metadata,
    file_paths_frame,
    frame_to_chunks,
    repo_refs_frame,
    scored_to_chunks,
    serial_map_frame,
)
from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind, QueryKind, RepoRef, ScoredChunk
from rbtr.index.reranker import Reranker
from rbtr.index.search import search
from rbtr.index.tokenise import tokenise_code
from rbtr.index.writer import WriteSession

log = structlog.get_logger(__name__)


# Pre-load all SQL at import time so file I/O is not on the hot path.
_GET_CHUNKS_SQL = load_sql("get_chunks.sql")
_GET_EDGES_SQL = load_sql("get_edges.sql")
_INBOUND_REFS_SQL = load_sql("inbound_refs.sql")
_DIFF_SYMBOLS_SQL = load_sql("diff_symbols.sql")
_SEARCH_BY_NAME_SQL = load_sql("search_by_name.sql")
_SEARCH_SIMILAR_SQL = load_sql("search_similar.sql")
_SEARCH_FULLTEXT_SQL = load_sql("search_fulltext.sql")
_COUNT_ORPHAN_CHUNKS_SQL = load_sql("count_orphan_chunks.sql")
_COUNT_GC_CHUNK_SPLIT_SQL = load_sql("count_gc_chunk_split.sql")
_INBOUND_DEGREE_SQL = load_sql("inbound_degree.sql")
_HAS_BLOB_SQL = load_sql("has_blob.sql")
_GET_SCHEMA_VERSION_SQL = load_sql("get_schema_version.sql")
_GET_REPO_SQL = load_sql("get_repo.sql")
_LIST_REPOS_SQL = load_sql("list_repos.sql")
_GET_CHUNK_PATHS_SQL = load_sql("get_chunk_paths.sql")
_COUNT_CHUNKS_SQL = load_sql("count_chunks.sql")
_HAS_INDEXED_SQL = load_sql("has_indexed.sql")
_LIST_INDEXED_COMMITS_SQL = load_sql("list_indexed_commits.sql")
_LIST_WATCHED_REFS_SQL = load_sql("list_watched_refs.sql")
_COUNT_SNAPSHOTS_FOR_COMMIT_SQL = load_sql("count_snapshots_for_commit.sql")
_COUNT_EDGES_FOR_COMMIT_SQL = load_sql("count_edges_for_commit.sql")
_GET_SNAPSHOT_LANGUAGE_SQL = load_sql("get_snapshot_language.sql")
_COUNT_UNEMBEDDED_SQL = load_sql("count_unembedded.sql")
_GET_UNEMBEDDED_CHUNKS_SQL = load_sql("get_unembedded_chunks.sql")


# ── Row mapping ──────────────────────────────────────────────────────

# ── Schema versioning ────────────────────────────────────────────────


def check_schema_version(db_path: Path) -> None:
    """Delete *db_path* if its schema version is stale.

    Returns silently if the file is missing or already open in
    this process.  Lets `duckdb.IOException` propagate when the
    DB is locked by another process or genuinely corrupt --
    callers decide what to do with that.
    """
    if not db_path.exists():
        return
    try:
        with contextlib.closing(duckdb.connect(str(db_path), read_only=True)) as con:
            rows = con.execute(_GET_SCHEMA_VERSION_SQL).fetchall()
    except (duckdb.ConnectionException, duckdb.IOException):
        return
    except (duckdb.CatalogException, duckdb.DependencyException):
        # CatalogException: meta table doesn't exist yet (fresh DB).
        # DependencyException: WAL contains stale FTS DDL that
        # can't replay in read-only mode.  The read-write open
        # in __init__ will handle it.
        rows = []

    stored = str(rows[0][0]) if rows else ""
    if stored == SCHEMA_VERSION:
        return
    log.warning(
        "index_schema_changed",
        stored=stored or "none",
        current=SCHEMA_VERSION,
    )
    db_path.unlink(missing_ok=True)
    db_path.with_suffix(db_path.suffix + ".wal").unlink(missing_ok=True)


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
            resolved = Path(db_path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            check_schema_version(resolved)
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
                raise RbtrError(locked_msg) from exc
            raise
        self._con.execute("INSTALL fts; LOAD fts;")
        self._local = threading.local()
        if writable:
            with self.session():
                pass  # schema DDL + embedding version check

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
        if cur is None:
            cur = self._con.cursor()
            self._local.cur = cur
        return cur

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

    def get_repo_id(self, path: str) -> int | None:
        """Return the repo_id for *path*, or None if not registered."""
        row = self._cursor.execute(_GET_REPO_SQL, {"path": path}).fetchone()
        return int(row[0]) if row else None

    def resolve_repo(self, repo: str) -> int:
        """Return the repo_id for *repo*, raising if not registered.

        Results are cached for the lifetime of the store.
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

    def list_repos(self) -> list[tuple[int, str]]:
        """Return all registered repos as `(id, path)` tuples."""
        rows = self._cursor.execute(_LIST_REPOS_SQL).fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    def list_watched_refs(self, repo_id: int) -> list[str]:
        """Return the repo's watched refs (symbolic names), sorted by name."""
        rows = self._cursor.execute(_LIST_WATCHED_REFS_SQL, {"repo_id": repo_id}).fetchall()
        return [str(r[0]) for r in rows]

    def latest_ref(self, repo_id: int, repo_path: str) -> RepoRef | None:
        """Resolve the most recent indexed ref for one repo.

        Prefers the current worktree tree SHA when the worktree is
        dirty and that tree has been indexed; otherwise falls back
        to the newest indexed commit.  Returns `None` when the repo
        has no indexed commits at all.
        """
        tree_sha = worktree_tree_sha(repo_path)
        if tree_sha is not None and self.has_indexed(repo_id, tree_sha):
            return RepoRef(repo_id=repo_id, commit_sha=tree_sha)
        indexed = self.list_indexed_commits(repo_id)
        if not indexed:
            return None
        return RepoRef(repo_id=repo_id, commit_sha=indexed[0][0])

    def list_latest_refs(self) -> list[RepoRef]:
        """Return one `RepoRef` per registered repo with indexed data.

        Applies `latest_ref` to every repo from `list_repos`,
        skipping repos that have never been indexed.  The basis for
        cross-repo search: the returned list is passed straight to
        `search()`.
        """
        refs: list[RepoRef] = []
        for repo_id, repo_path in self.list_repos():
            ref = self.latest_ref(repo_id, repo_path)
            if ref is not None:
                refs.append(ref)
        return refs

    # ── Completion tracking (indexed_commits) ──────────────────────

    def has_indexed(self, repo_id: int, commit_sha: str) -> bool:
        """Return whether *commit_sha* was fully indexed."""
        row = self._cursor.execute(
            _HAS_INDEXED_SQL, {"repo_id": repo_id, "commit_sha": commit_sha}
        ).fetchone()
        return row is not None

    def list_indexed_commits(self, repo_id: int) -> list[tuple[str, str]]:
        """Return `(commit_sha, indexed_at)` for this repo, newest first."""
        rows = self._cursor.execute(_LIST_INDEXED_COMMITS_SQL, {"repo_id": repo_id}).fetchall()
        return [
            (
                str(r[0]),
                str(r[1]),
            )
            for r in rows
        ]

    def count_snapshots_for_commit(self, repo_id: int, commit_sha: str) -> int:
        """Return the number of `file_snapshots` rows for this commit.

        Read-only. Used by dry-run GC reporting.
        """
        row = self._cursor.execute(
            _COUNT_SNAPSHOTS_FOR_COMMIT_SQL, {"repo_id": repo_id, "commit_sha": commit_sha}
        ).fetchone()
        return int(row[0]) if row else 0

    def count_edges_for_commit(self, repo_id: int, commit_sha: str) -> int:
        """Return the number of `edges` rows for this commit.

        Read-only. Used by dry-run GC reporting.
        """
        row = self._cursor.execute(
            _COUNT_EDGES_FOR_COMMIT_SQL,
            {"repo_id": repo_id, "commit_sha": commit_sha},
        ).fetchone()
        return int(row[0]) if row else 0

    def count_orphan_chunks(self) -> int:
        """Count chunks not referenced by any file snapshot in any repo.

        The chunk store is content-addressed and shared, so orphan
        status is global: a chunk is an orphan iff no snapshot
        anywhere references its `(blob_sha, file_path)`.
        """
        row = self._cursor.execute(_COUNT_ORPHAN_CHUNKS_SQL).fetchone()
        return int(row[0]) if row else 0

    def count_gc_chunk_split(self, repo_id: int, drop_shas: list[str]) -> tuple[int, int]:
        """Split a GC drop set's chunks into `(dropped, kept_shared)`.

        *dropped* is the chunks the drop set would free; *kept_shared*
        is candidate chunks retained because a ref outside the drop set
        (another ref of this repo, or any other repo) still references
        their `(blob_sha, file_path)`.  Read-only — it computes the
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

    def has_blob(self, blob_sha: str, language: str, serials: dict[str, int]) -> bool:
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

        When `has_blob` returns False and old chunks exist for
        the blob (language change), the caller must delete old
        chunks before inserting new ones — the new extraction
        may produce different chunk IDs that the upsert can't
        reconcile.
        """
        self._cursor.register("_serial_map", serial_map_frame(serials))
        try:
            row = self._cursor.execute(
                _HAS_BLOB_SQL, {"blob_sha": blob_sha, "language": language}
            ).fetchone()
        finally:
            self._cursor.unregister("_serial_map")
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

    def count_unembedded(self, repo_id: int, commit_sha: str) -> int:
        """Count chunks visible at *commit_sha* that lack embeddings."""
        row = self._cursor.execute(
            _COUNT_UNEMBEDDED_SQL, {"repo_id": repo_id, "commit_sha": commit_sha}
        ).fetchone()
        return int(row[0]) if row else 0

    def get_unembedded_chunks(
        self, repo_id: int, commit_sha: str, limit: int = 1000
    ) -> list[Chunk]:
        """Return chunks at *commit_sha* with `embedding IS NULL`.

        Results are ordered deterministically by `(file_path, line_start)`
        and capped at *limit*.
        """
        params = {
            "repo_id": repo_id,
            "commit_sha": commit_sha,
            "max_rows": limit,
        }
        frame = (
            self._cursor.execute(_GET_UNEMBEDDED_CHUNKS_SQL, params)
            .pl()
            .pipe(_decode_metadata)
            .pipe(ChunkResultRow.validate, cast=True)
        )
        return frame_to_chunks(frame)

    def count_chunks(self, commit_sha: str, repo_id: int) -> int:
        """Count chunks visible at *commit_sha* without loading them."""
        row = self._cursor.execute(
            _COUNT_CHUNKS_SQL, {"repo_id": repo_id, "commit_sha": commit_sha}
        ).fetchone()
        return int(row[0]) if row else 0

    def get_chunks(
        self,
        commit_sha: str,
        *,
        file_path: str | None = None,
        kind: ChunkKind | None = None,
        name: str | None = None,
        repo_id: int,
    ) -> list[Chunk]:
        """Query chunks visible at *commit_sha* with optional filters."""
        kind_val = kind.value if kind is not None else None
        params = {
            "repo_id": repo_id,
            "commit_sha": commit_sha,
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
        self, commit_sha: str, target_ids: list[str], *, repo_id: int
    ) -> dy.DataFrame[InboundRefResultRow]:
        """Return referrers of the given target chunks at *commit_sha*.

        One row per inbound edge, resolved to the source (referrer)
        chunk's identity plus the edge kind — powers `find-refs`.
        """
        if not target_ids:
            return InboundRefResultRow.create_empty()
        self._cursor.register(
            "_repo_refs", repo_refs_frame([RepoRef(repo_id=repo_id, commit_sha=commit_sha)])
        )
        try:
            return (
                self._cursor.execute(_INBOUND_REFS_SQL, {"target_ids": target_ids})
                .pl()
                .pipe(InboundRefResultRow.validate, cast=True)
            )
        finally:
            self._cursor.unregister("_repo_refs")

    def get_chunks_frame(self, commit_sha: str, *, repo_id: int) -> dy.DataFrame[ChunkContentRow]:
        """Return all chunks at *commit_sha* as a content-only frame.

        The frame is validated through `ChunkContentRow` and
        contains identity columns (`file_path`, `scope`,
        `name`, `line_start`) plus `language` and `content`.
        """
        params = {
            "repo_id": repo_id,
            "commit_sha": commit_sha,
            "file_path": None,
            "kind": None,
            "name": None,
        }
        return (
            self._cursor.execute(_GET_CHUNKS_SQL, params)
            .pl()
            .select("file_path", "scope", "name", "line_start", "language", "content")
            .pipe(ChunkContentRow.validate, cast=True)
        )

    def _get_edges_frame(
        self,
        refs: list[RepoRef],
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        kind: EdgeKind | None = None,
    ) -> dy.DataFrame[EdgeResultRow]:
        """Return edges scoped to *refs* as a validated frame."""
        kind_val = kind.value if kind is not None else None
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "kind": kind_val,
        }
        self._cursor.register("_repo_refs", repo_refs_frame(refs))
        try:
            return (
                self._cursor.execute(_GET_EDGES_SQL, params)
                .pl()
                .pipe(EdgeResultRow.validate, cast=True)
            )
        finally:
            self._cursor.unregister("_repo_refs")

    def get_edges(
        self,
        commit_sha: str,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        kind: EdgeKind | None = None,
        repo_id: int,
    ) -> list[Edge]:
        """Query edges scoped to *commit_sha*."""
        frame = self._get_edges_frame(
            [RepoRef(repo_id=repo_id, commit_sha=commit_sha)],
            source_id=source_id,
            target_id=target_id,
            kind=kind,
        )
        return [
            Edge(
                source_id=row["source_id"],
                target_id=row["target_id"],
                kind=EdgeKind(row["kind"]),
            )
            for row in frame.iter_rows(named=True)
        ]

    def inbound_degrees(
        self, refs: list[RepoRef], chunk_ids: list[str]
    ) -> dy.DataFrame[InboundDegreeResultRow]:
        """Return inbound edge counts for the given chunk IDs."""
        if not chunk_ids:
            return InboundDegreeResultRow.create_empty()
        self._cursor.register("_repo_refs", repo_refs_frame(refs))
        try:
            return (
                self._cursor.execute(
                    _INBOUND_DEGREE_SQL,
                    {"chunk_ids": chunk_ids},
                )
                .pl()
                .pipe(InboundDegreeResultRow.validate, cast=True)
            )
        finally:
            self._cursor.unregister("_repo_refs")

    def diff_symbols(
        self,
        base_sha: str,
        head_sha: str,
        *,
        repo_id: int,
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
        in `diff_symbols.sql`; `None` or an empty list diffs every
        file (the `scope_all` flag bypasses the view).
        """
        params = {
            "repo_id": repo_id,
            "head_sha": head_sha,
            "base_sha": base_sha,
            "scope_all": not file_paths,
        }
        self._cursor.register("_file_paths", file_paths_frame(file_paths or []))
        try:
            return (
                self._cursor.execute(_DIFF_SYMBOLS_SQL, params)
                .pl()
                .pipe(_decode_metadata)
                .pipe(ChangedSymbolRow.validate, cast=True)
            )
        finally:
            self._cursor.unregister("_file_paths")

    # ── Match (internal frame, public chunk) ─────────────────────

    def _match_by_name(self, refs: list[RepoRef], pattern: str) -> dy.DataFrame[ChunkResultRow]:
        """Return name-matched chunks as a validated frame.

        Resolution is tiered: exact → case-insensitive exact →
        prefix → substring.  Only the best tier that has matches
        is returned.
        """
        self._cursor.register("_repo_refs", repo_refs_frame(refs))
        try:
            return (
                self._cursor.execute(
                    _SEARCH_BY_NAME_SQL,
                    {"name": pattern, "pattern": f"%{pattern}%"},
                )
                .pl()
                .pipe(_decode_metadata)
                .pipe(ChunkResultRow.validate, cast=True)
            )
        finally:
            self._cursor.unregister("_repo_refs")

    def match_by_name(self, commit_sha: str, pattern: str, *, repo_id: int) -> list[Chunk]:
        """Find chunks by name with tiered resolution.

        Prefers exact matches, then case-insensitive exact, then
        prefix, then substring.  Returns only the best tier.
        """
        return frame_to_chunks(
            self._match_by_name([RepoRef(repo_id=repo_id, commit_sha=commit_sha)], pattern)
        )

    def _match_similar(
        self,
        refs: list[RepoRef],
        query_embeddings: list[list[float]],
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
        self._cursor.register("_qvecs", vecs_frame)
        self._cursor.register("_repo_refs", repo_refs_frame(refs))
        try:
            return (
                self._cursor.execute(
                    _SEARCH_SIMILAR_SQL,
                    {"top_k": top_k},
                )
                .pl()
                .pipe(_decode_metadata)
                .pipe(ScoredChunkResultRow.validate, cast=True)
            )
        finally:
            self._cursor.unregister("_qvecs")
            self._cursor.unregister("_repo_refs")

    def match_similar(
        self,
        commit_sha: str,
        query_embedding: list[float],
        top_k: int = 10,
        *,
        repo_id: int,
    ) -> list[tuple[Chunk, float]]:
        """Find the *top_k* chunks most similar to *query_embedding*."""
        return scored_to_chunks(
            self._match_similar(
                [RepoRef(repo_id=repo_id, commit_sha=commit_sha)], [query_embedding], top_k
            )
        )

    def _match_by_text(
        self,
        commit_sha: str,
        query: str,
        top_k: int = 10,
        *,
        repo_id: int,
        embedder: Embedder | None = None,
    ) -> dy.DataFrame[ScoredChunkResultRow]:
        """Embed *query* and return similar chunks as a scored frame."""
        if embedder is None:
            return ScoredChunkResultRow.create_empty()
        prefix = config.query_instruction
        text = f"{prefix}{query}" if prefix else query
        query_embedding = embedder.embed_single(text)
        return self._match_similar(
            [RepoRef(repo_id=repo_id, commit_sha=commit_sha)], [query_embedding], top_k
        )

    def match_by_text(
        self,
        commit_sha: str,
        query: str,
        top_k: int = 10,
        *,
        repo_id: int,
        embedder: Embedder | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Semantic search: embed *query* then find similar chunks."""
        return scored_to_chunks(
            self._match_by_text(commit_sha, query, top_k, repo_id=repo_id, embedder=embedder)
        )

    # ── FTS ──────────────────────────────────────────────────────────

    def _match_fulltext(
        self,
        refs: list[RepoRef],
        query: str,
        top_k: int = 10,
    ) -> dy.DataFrame[ScoredChunkResultRow]:
        """Return BM25-matched chunks as a validated scored frame."""
        tokenised_query = tokenise_code(query)
        if not tokenised_query:
            return ScoredChunkResultRow.create_empty()
        self._cursor.register("_repo_refs", repo_refs_frame(refs))
        try:
            return (
                self._cursor.execute(
                    _SEARCH_FULLTEXT_SQL,
                    {
                        "tokenised_query": tokenised_query,
                        "top_k": top_k,
                    },
                )
                .pl()
                .pipe(_decode_metadata)
                .pipe(ScoredChunkResultRow.validate, cast=True)
            )
        except duckdb.CatalogException as exc:
            raise IndexNotBuiltError from exc
        finally:
            self._cursor.unregister("_repo_refs")

    def match_fulltext(
        self,
        commit_sha: str,
        query: str,
        top_k: int = 10,
        *,
        repo_id: int,
    ) -> list[tuple[Chunk, float]]:
        """BM25 keyword search across chunk name and content.

        The *query* is pre-tokenised with `tokenise_code` so
        that identifier queries (`AgentDeps` -> `agentdeps agent
        deps`) match the code-aware tokens stored in the index.

        Raises `IndexNotBuiltError` if no FTS index exists.
        """
        return scored_to_chunks(
            self._match_fulltext([RepoRef(repo_id=repo_id, commit_sha=commit_sha)], query, top_k)
        )

    def _fetch_chunk_paths(
        self, refs: list[RepoRef], chunk_ids: list[str]
    ) -> dy.DataFrame[ChunkPathResultRow]:
        """Return `(id, file_path)` for the given chunk IDs."""
        if not chunk_ids:
            return ChunkPathResultRow.create_empty()
        self._cursor.register("_repo_refs", repo_refs_frame(refs))
        try:
            return (
                self._cursor.execute(
                    _GET_CHUNK_PATHS_SQL,
                    {"chunk_ids": chunk_ids},
                )
                .pl()
                .pipe(ChunkPathResultRow.validate, cast=True)
            )
        finally:
            self._cursor.unregister("_repo_refs")

    # ── Unified search ───────────────────────────────────────────────

    def search(
        self,
        refs: list[RepoRef],
        query: str,
        *,
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
        details.  A one-element *refs* list is a single-repo
        search; many refs fan the query across repos.  *repo_paths*
        maps `repo_id` to a path so cross-repo results carry their
        origin.
        """
        return search(
            self,
            refs,
            query,
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
