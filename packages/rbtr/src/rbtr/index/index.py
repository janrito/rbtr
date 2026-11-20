"""The code index's API: registry changes and every read.

Writes go through the `WriteSession` that `Index.session` returns.
ARCHITECTURE's "Data model" describes the files and tables.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import dataframely as dy
import duckdb
import polars as pl
import structlog

from rbtr.config import config
from rbtr.domain.models import (
    Chunk,
    ChunkKind,
    ChunkSplit,
    EdgeKind,
    IndexedSnapshot,
    Repo,
    SnapshotCounts,
    SnapshotRange,
    SnapshotRef,
)
from rbtr.domain.tokenise import tokenise_code
from rbtr.errors import (
    RepoNotFoundError,
)
from rbtr.index import load_sql
from rbtr.index.db import registered_views
from rbtr.index.results import (
    ChangedSymbolRow,
    ChunkContentRow,
    ChunkPathResultRow,
    ChunkResultRow,
    EdgeResultRow,
    InboundDegreeResultRow,
    InboundRefResultRow,
    NameMatchRow,
    ScoredChunkResultRow,
    SnapshotCountsRow,
    _decode_metadata,
    chunk_ids_view,
    file_paths_view,
    frame_to_chunks,
    frame_to_snapshot_counts,
    snapshot_refs_view,
)
from rbtr.index.store import Store
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
_GET_CHUNK_PATHS_SQL = load_sql("get_chunk_paths.sql")
_CHUNK_COUNTS_BY_SNAPSHOT_SQL = load_sql("chunk_counts_by_snapshot.sql")
_DISTINCT_CHUNK_LANGUAGES_SQL = load_sql("distinct_chunk_languages.sql")
_HAS_INDEXED_SQL = load_sql("has_indexed.sql")
_LIST_INDEXED_COMMITS_SQL = load_sql("list_indexed_snapshots.sql")
_LIST_WATCHED_REFS_SQL = load_sql("list_watched_refs.sql")
_COUNT_FILE_SNAPSHOTS_SQL = load_sql("count_file_snapshots.sql")
_COUNT_EDGES_SQL = load_sql("count_edges.sql")
_UNEMBEDDED_CHUNK_IDS_SQL = load_sql("unembedded_chunk_ids.sql")
_GET_CHUNKS_BY_ID_SQL = load_sql("get_chunks_by_id.sql")


class Index:
    """Synchronous storage access for reads, registry changes and write sessions.

    Reads and registry methods use thread-local cursors and may run on different
    threads. Calls block, so keep them off a UI loop. The caller coordinates
    overlapping write workflows; Index methods do not acquire a shared writer
    lock. Create and use each `WriteSession` on one thread.

    Read-only by default: registry writes and `session` require `writable=True`
    or raise `RuntimeError`. The caller owns the handle and closes it after all
    users finish.
    """

    def __init__(self, db_path: Path | str | None = None, *, writable: bool = False) -> None:
        self._store = Store(db_path, writable=writable)

    @property
    def db_path(self) -> str | None:
        """The registry's path; `None` for an in-memory index."""
        return self._store.db_path

    @classmethod
    def from_config(cls, *, writable: bool = False) -> Index:
        """Open the database at the central DB path."""
        return cls(config.db_path, writable=writable)

    @contextmanager
    def reader(self, repo: Repo, **frames: pl.DataFrame) -> Iterator[duckdb.DuckDBPyConnection]:
        """A cursor on *repo*'s project file, with *frames* registered as temp views.

        The query a caller writes decides what it reads; *frames* bind the
        values it joins against, each under its keyword name, and are
        dropped when the block ends.
        """
        with registered_views(self._store.data(repo), **frames) as cur:
            yield cur

    def session(self, repo: Repo) -> WriteSession:
        """Return a transaction context manager for rows belonging to `repo`.

        Create and use it on the same thread. Entry begins a transaction; normal
        exit flushes and commits, exceptional exit rolls back that transaction.
        Writes must satisfy `WriteSession`'s completion rules. Raise `RuntimeError`
        on a read-only Index.
        """
        self._store.require_writable()
        return WriteSession(self._store.data(repo), repo)

    def close(self) -> None:
        """Delete retired files and close connections after all users have finished."""
        self._store.close()

    def retire_due(self) -> None:
        """Detach and delete every retired file whose delay has passed."""
        self._store.retire_due()

    def disk_size_bytes(self) -> int:
        """Total bytes the index occupies on disk; see `Store.disk_size_bytes`."""
        return self._store.disk_size_bytes()

    def register(self, path: str) -> Repo:
        """Register a repo path and return the repo; see `Store.register`."""
        return self._store.register(path)

    def forget(self, repo: Repo) -> None:
        """Forget a registered repo; see `Store.forget`."""
        self._store.forget(repo)

    def compact(self, repo: Repo) -> None:
        """Rewrite *repo*'s project file into a fresh one; see `Store.compact`."""
        self._store.compact(repo)

    def compact_all(self) -> None:
        """Rewrite every project file into a fresh one; see `Store.compact`."""
        self._store.compact_all()

    def compact_grown(self, ratio: float) -> int:
        """Compact every project file grown past *ratio*; see `Store.compact_grown`."""
        return self._store.compact_grown(ratio)

    def missing_since(self) -> dict[Repo, datetime]:
        """When each repo whose path was found gone was first found so."""
        return self._store.missing_since()

    def set_missing_since(self, repo: Repo, when: datetime | None) -> None:
        """Record when *repo*'s path was first found gone; `None` clears it."""
        self._store.set_missing_since(repo, when)

    def find_repo(self, path: str) -> Repo | None:
        """Return the repo registered at *path*, or None if there is none.

        *path* is matched exactly, as stored at registration — it is not
        normalised, so pass a canonical path.  A repo whose checkout has
        since been deleted is still found, which is how it gets forgotten.
        """
        return self._store.repo_at(path)

    def repo(self, path: str) -> Repo:
        """Return the repo registered at *path*, raising if there is none.

        As with `find_repo`, *path* is matched exactly and must already
        be canonical.
        """
        found = self._store.repo_at(path)
        if found is None:
            msg = f"Repo not registered: {path}"
            raise RepoNotFoundError(msg)
        return found

    def repos(self) -> list[Repo]:
        """Return all registered repos, in the order they were registered."""
        return self._store.repos()

    def watched_refs(self, repo: Repo) -> list[str]:
        """Return the repo's watched refs (symbolic names), sorted by name."""
        rows = (
            self._store.data(repo)
            .sql(_LIST_WATCHED_REFS_SQL, params={"repo_id": repo.repo_id})
            .fetchall()
        )
        return [str(r[0]) for r in rows]

    def has_indexed(self, *, at: SnapshotRef) -> bool:
        """Return whether the snapshot at *at* was fully indexed."""
        rows = (
            self._store.data(at.repo)
            .sql(
                _HAS_INDEXED_SQL,
                params={"repo_id": at.repo.repo_id, "snapshot_sha": at.snapshot_sha},
            )
            .fetchall()
        )
        return bool(rows)

    def indexed_snapshots(self, repo: Repo) -> list[IndexedSnapshot]:
        """Return the repo's indexed snapshots, newest first."""
        rows = (
            self._store.data(repo)
            .sql(_LIST_INDEXED_COMMITS_SQL, params={"repo_id": repo.repo_id})
            .fetchall()
        )
        return [IndexedSnapshot(snapshot_sha=str(r[0]), indexed_at=r[1]) for r in rows]

    def count_file_snapshots(self, *, at: SnapshotRef) -> int:
        """Return the number of `file_snapshots` rows for this commit.

        Read-only. Used by dry-run GC reporting.
        """
        rows = (
            self._store.data(at.repo)
            .sql(
                _COUNT_FILE_SNAPSHOTS_SQL,
                params={"repo_id": at.repo.repo_id, "snapshot_sha": at.snapshot_sha},
            )
            .fetchall()
        )
        return int(rows[0][0]) if rows else 0

    def count_edges(self, *, at: SnapshotRef) -> int:
        """Return the number of `edges` rows for this commit.

        Read-only. Used by dry-run GC reporting.
        """
        rows = (
            self._store.data(at.repo)
            .sql(
                _COUNT_EDGES_SQL,
                params={"repo_id": at.repo.repo_id, "snapshot_sha": at.snapshot_sha},
            )
            .fetchall()
        )
        return int(rows[0][0]) if rows else 0

    def count_orphan_chunks(self) -> int:
        """Count chunks no file snapshot in their project references.

        A project's chunks are content-addressed and shared by its
        checkouts, so a chunk is an orphan iff no snapshot of any of them
        references its `(blob_sha, file_language)`.  Summed over projects.
        """
        rows = self._store.each_file(lambda cur: cur.sql(_COUNT_ORPHAN_CHUNKS_SQL).fetchall())
        return sum(int(file_rows[0][0]) for file_rows in rows if file_rows)

    def count_gc_chunk_split(self, repo: Repo, drop_shas: list[str]) -> ChunkSplit:
        """Split a GC drop set's chunks into those it frees and those it keeps.

        `dropped` is the chunks the drop set would free; `kept` is
        candidate chunks retained because a ref outside the drop set
        (another ref of this repo, or another checkout of its project) still references
        their `(blob_sha, file_language)`.  Read-only — it computes the
        split from the reference graph without simulating the drop, so
        it serves dry-run and real runs alike.
        """
        if not drop_shas:
            return ChunkSplit(dropped=0, kept=0)
        params = {"repo_id": repo.repo_id, "drop_shas": drop_shas}
        rows = self._store.data(repo).sql(_COUNT_GC_CHUNK_SPLIT_SQL, params=params).fetchall()
        if not rows:
            return ChunkSplit(dropped=0, kept=0)
        return ChunkSplit(dropped=int(rows[0][0]), kept=int(rows[0][1]))

    def chunk_counts_by_snapshot(
        self, *, repo: Repo | None = None
    ) -> dict[SnapshotRef, SnapshotCounts]:
        """Return chunk and embedded counts per indexed snapshot.

        Spans every repo unless *repo* narrows it.  One grouped pass
        whatever the scope, so the cost does not grow with the number of
        snapshots asked about — which matters because reading
        `chunks.embedding` is most of the work.

        Every snapshot in `indexed_snapshots` gets an entry; one holding no
        chunks counts zero.  Ordered most recently indexed first, ties
        broken by `snapshot_sha`.
        """
        frame = self._chunk_counts_frame(repo)
        return frame_to_snapshot_counts(frame, {r.repo_id: r for r in self._store.repos()})

    def _chunk_counts_frame(
        self, scope: Repo | SnapshotRef | None
    ) -> dy.DataFrame[SnapshotCountsRow]:
        """Back the counts readers with one validated frame.

        *scope* narrows the count to one repo or one of its snapshots.
        `None` counts every project file and merges the frames in order.
        """
        repo = scope.repo if isinstance(scope, SnapshotRef) else scope
        snapshot_sha = scope.snapshot_sha if isinstance(scope, SnapshotRef) else None
        repo_id = repo.repo_id if repo is not None else None
        params = {"repo_id": repo_id, "snapshot_sha": snapshot_sha}

        def count(cur: duckdb.DuckDBPyConnection) -> pl.DataFrame:
            return cur.sql(_CHUNK_COUNTS_BY_SNAPSHOT_SQL, params=params).pl()

        frames = (
            [count(self._store.data(repo))] if repo is not None else self._store.each_file(count)
        )
        if not frames:
            return SnapshotCountsRow.create_empty()
        return (
            pl.concat(frames)
            .sort(["indexed_at", "snapshot_sha"], descending=[True, False])
            .pipe(SnapshotCountsRow.validate, cast=True)
        )

    def chunk_counts_for_snapshot(self, *, at: SnapshotRef) -> SnapshotCounts:
        """Return chunk and embedded counts for one indexed snapshot.

        Counts chunks, so content held at several paths counts once — the
        figure embedding progress is measured against.  A snapshot that
        holds no chunks, or was never marked indexed, reads as zero of
        zero.
        """
        frame = self._chunk_counts_frame(at)
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
        rows = (
            self._store.data(at.repo)
            .sql(
                _UNEMBEDDED_CHUNK_IDS_SQL,
                params={"repo_id": at.repo.repo_id, "snapshot_sha": at.snapshot_sha},
            )
            .fetchall()
        )
        return [str(r[0]) for r in rows]

    def chunks_by_id(self, chunk_ids: list[str], *, at: SnapshotRef) -> list[Chunk]:
        """Return the named chunks as they are seen at *at*, ordered by id.

        An id that no longer resolves at *at* is left out of the
        result, which shortens the page: a chunk can be collected
        between a work list being drawn up and a page of it being read.
        """
        if not chunk_ids:
            return []
        params = {"repo_id": at.repo.repo_id, "snapshot_sha": at.snapshot_sha}
        with registered_views(
            self._store.data(at.repo), _chunk_ids=chunk_ids_view(chunk_ids)
        ) as cur:
            frame = (
                cur.sql(_GET_CHUNKS_BY_ID_SQL, params=params)
                .pl()
                .pipe(_decode_metadata)
                .pipe(ChunkResultRow.validate, cast=True)
            )
        return frame_to_chunks(frame)

    def distinct_chunk_languages(self) -> set[str]:
        """Return every real language present in any project's file.

        Excludes the raw-chunk fallback pseudo-language (`''`), so the
        result is the set of languages some plugin has actually
        extracted -- a record of the plugin set that built the index.
        """
        rows = self._store.each_file(lambda cur: cur.sql(_DISTINCT_CHUNK_LANGUAGES_SQL).fetchall())
        return {str(r[0]) for file_rows in rows for r in file_rows}

    def chunks(
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
            "repo_id": at.repo.repo_id,
            "snapshot_sha": at.snapshot_sha,
            "file_path": file_path,
            "kind": kind_val,
            "name": name,
        }
        frame = (
            self._store.data(at.repo)
            .sql(_GET_CHUNKS_SQL, params=params)
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
        with registered_views(
            self._store.data(at.repo), _snapshot_refs=snapshot_refs_view([at])
        ) as cur:
            return (
                cur.sql(_INBOUND_REFS_SQL, params={"target_ids": target_ids})
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
            "repo_id": at.repo.repo_id,
            "snapshot_sha": at.snapshot_sha,
            "file_path": None,
            "kind": None,
            "name": None,
        }
        return (
            self._store.data(at.repo)
            .sql(_GET_CHUNKS_SQL, params=params)
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
        return _concat(self._within(within, _GET_EDGES_SQL, params), EdgeResultRow)

    def inbound_degrees(
        self, chunk_ids: list[str], *, within: list[SnapshotRef]
    ) -> dy.DataFrame[InboundDegreeResultRow]:
        """Return inbound edge counts for the given chunk IDs.

        A chunk two projects share has edges in each file; its degree is
        their sum.
        """
        if not chunk_ids:
            return InboundDegreeResultRow.create_empty()
        frames = self._within(within, _INBOUND_DEGREE_SQL, {"chunk_ids": chunk_ids})
        if not frames:
            return InboundDegreeResultRow.create_empty()
        return (
            pl.concat(frames)
            .group_by("chunk_id")
            .agg(pl.col("degree").sum())
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
            "repo_id": between.repo.repo_id,
            "head_sha": between.head_sha,
            "base_sha": between.base_sha,
            "scope_all": not file_paths,
        }
        paths = file_paths_view(file_paths or [])
        with registered_views(self._store.data(between.repo), _file_paths=paths) as cur:
            return (
                cur.sql(_CHANGED_SYMBOLS_SQL, params=params)
                .pl()
                .pipe(_decode_metadata)
                .pipe(ChangedSymbolRow.validate, cast=True)
            )

    def name_matches(
        self, pattern: str, *, within: list[SnapshotRef]
    ) -> dy.DataFrame[ChunkResultRow]:
        """Return name-matched chunks as a validated frame.

        Resolution is tiered: exact → case-insensitive exact →
        prefix → substring.  Only the best tier that has matches
        is returned, across every file *within* spans.
        """
        params = {"name": pattern, "pattern": f"%{pattern}%"}
        frames = self._within(within, _SEARCH_BY_NAME_SQL, params)
        if not frames:
            return ChunkResultRow.create_empty()
        matches = pl.concat(frames).pipe(_decode_metadata).pipe(NameMatchRow.validate, cast=True)
        return (
            matches.filter(pl.col("match_tier") == pl.col("match_tier").min())
            .drop("match_tier")
            .sort("file_path", "line_start")
            .pipe(ChunkResultRow.validate, cast=True)
        )

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
        Each file returns its best *top_k*; the merge keeps the best
        *top_k* of those.
        """
        vecs_frame = pl.DataFrame({"vec": query_embeddings}).cast({"vec": pl.List(pl.Float32)})
        frames = self._within(within, _SEARCH_SIMILAR_SQL, {"top_k": top_k}, _qvecs=vecs_frame)
        return _best(frames, top_k)

    def fulltext_matches(
        self,
        query: str,
        *,
        within: list[SnapshotRef],
        top_k: int = 10,
    ) -> dy.DataFrame[ScoredChunkResultRow]:
        """Return BM25-matched chunks as a validated scored frame.

        Each file scores against its own statistics and returns its best
        *top_k*; the merge keeps the best *top_k* of those.
        """
        tokenised_query = tokenise_code(query)
        if not tokenised_query:
            return ScoredChunkResultRow.create_empty()
        params = {"tokenised_query": tokenised_query, "top_k": top_k}
        return _best(self._within(within, _SEARCH_FULLTEXT_SQL, params), top_k)

    def chunk_paths(
        self, chunk_ids: list[str], *, within: list[SnapshotRef]
    ) -> dy.DataFrame[ChunkPathResultRow]:
        """Return `(id, file_path)` for the given chunk IDs."""
        if not chunk_ids:
            return ChunkPathResultRow.create_empty()
        frames = self._within(within, _GET_CHUNK_PATHS_SQL, {"chunk_ids": chunk_ids})
        return _concat(frames, ChunkPathResultRow)

    def _within(
        self,
        within: list[SnapshotRef],
        sql: str,
        params: Mapping[str, object],
        **frames: pl.DataFrame,
    ) -> list[pl.DataFrame]:
        """Run *sql* on each file *within* spans, with its refs as `_snapshot_refs`.

        *frames* are registered beside the refs.  One raw frame per file.
        """

        def read(cur: duckdb.DuckDBPyConnection, refs: list[SnapshotRef]) -> pl.DataFrame:
            views = {"_snapshot_refs": snapshot_refs_view(refs), **frames}
            with registered_views(cur, **views) as c:
                return c.sql(sql, params=params).pl()

        return self._store.across(within, read)


def _concat[S: dy.Schema](frames: list[pl.DataFrame], schema: type[S]) -> dy.DataFrame[S]:
    """One validated frame of the per-file *frames*."""
    if not frames:
        return schema.create_empty()
    return pl.concat(frames).pipe(schema.validate, cast=True)


def _best(frames: list[pl.DataFrame], top_k: int) -> dy.DataFrame[ScoredChunkResultRow]:
    """The *top_k* best-scored rows of the per-file *frames*."""
    if not frames:
        return ScoredChunkResultRow.create_empty()
    return (
        pl.concat(frames)
        .sort("score", descending=True)
        .head(top_k)
        .pipe(_decode_metadata)
        .pipe(ScoredChunkResultRow.validate, cast=True)
    )
