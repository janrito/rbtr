"""The index's database file: its connection, schema, registry, cursors, and compaction.

Only `Index` uses `Store`.  ARCHITECTURE's "Concurrency model"
describes how readers share it.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from uuid import uuid4

import duckdb

# duckdb requires this imported before any thread uses a connection.
import pyarrow  # noqa: F401
import structlog

from rbtr.config import config, own_dir
from rbtr.domain.models import Repo
from rbtr.errors import IndexLockedError, IndexNotBuiltError, IndexSchemaTooNewError, RbtrError
from rbtr.index import load_sql
from rbtr.index.constants import EMBEDDING_FORMAT_VERSION, SCHEMA_VERSION

log = structlog.get_logger(__name__)

_SCHEMA_SQL = load_sql("schema.sql")
_GET_SCHEMA_VERSION_SQL = load_sql("get_schema_version.sql")
_SET_SCHEMA_VERSION_SQL = load_sql("set_schema_version.sql")
_GET_EMBEDDING_META_SQL = load_sql("get_embedding_meta.sql")
_COUNT_EMBEDDINGS_SQL = load_sql("count_embeddings.sql")
_CLEAR_ALL_EMBEDDINGS_SQL = load_sql("clear_all_embeddings.sql")
_SET_EMBEDDING_VERSION_SQL = load_sql("set_embedding_version.sql")
_SET_EMBEDDING_MODEL_SQL = load_sql("set_embedding_model.sql")
_HAS_FTS_INDEX_SQL = load_sql("has_fts_index.sql")
_DROP_FTS_INDEX_SQL = load_sql("drop_fts_index.sql")
_WIPE_SCHEMA_SQL = load_sql("wipe_schema.sql")
_DATA_SIZE_BYTES_SQL = load_sql("data_size_bytes.sql")
_GET_REPO_SQL = load_sql("get_repo.sql")
_LIST_REPOS_SQL = load_sql("list_repos.sql")
_REGISTER_REPO_SQL = load_sql("register_repo.sql")
_FORGET_REPO_SQL = load_sql("delete_repo.sql")
# A repo's rows, deleted when it is forgotten.
_FORGET_ROWS_SQL = [
    load_sql("delete_snapshots_for_repo.sql"),
    load_sql("delete_edges_for_repo.sql"),
    load_sql("delete_indexed_snapshots_for_repo.sql"),
    load_sql("delete_watched_refs_for_repo.sql"),
]


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


class Store:
    """The index's DuckDB database file, open in this process.

    Holding the connection is owning the index: DuckDB's file lock keeps
    every other process out.  Opening checks the stored schema version
    and, when writable, creates the schema.  Registering and forgetting a
    repo are its methods.  Methods may be called from
    any thread: each thread gets its own cursor.
    """

    def __init__(self, db_path: Path | str | None = None, *, writable: bool = False) -> None:
        self.db_path: str | None = str(db_path) if db_path else None
        self._writable = writable
        if db_path is not None:
            own_dir(Path(db_path).parent)
        dsn = str(db_path) if db_path else ":memory:"
        try:
            self._con = self._connect(dsn)
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
            self._bootstrap()

    def require_writable(self) -> None:
        """Raise `RuntimeError` unless this process may change the index."""
        if not self._writable:
            msg = "Writing to the index requires a writable Index (pass writable=True)"
            raise RuntimeError(msg)

    # ── Registry ─────────────────────────────────────────────────────

    def register(self, path: str) -> Repo:
        """Register a repo path and return the repo.

        Idempotent — returns the existing repo if already registered.
        *path* must be canonical, since it is what identifies the repo:
        two spellings of one checkout would become two repos.  Pass it
        through `normalise_repo_path` first, as `build_index` does.
        """
        self.require_writable()
        existing = self.repo_at(path)
        if existing is not None:
            return existing
        row = self.cursor().execute(_REGISTER_REPO_SQL, {"path": path}).fetchall()
        return Repo(repo_id=int(row[0][0]), repo_path=path)

    def forget(self, repo: Repo) -> None:
        """Delete a registered repo and every row it owns, in one transaction.

        Its `file_snapshots`, `edges`, `indexed_snapshots` and
        `watched_refs` go with its `repos` row.  Its chunks stay: other
        repos may share them, so reclaiming the ones it orphans is GC's
        job.
        """
        self.require_writable()
        cur = self.cursor()
        cur.begin()
        for sql in (*_FORGET_ROWS_SQL, _FORGET_REPO_SQL):
            cur.execute(sql, {"repo_id": repo.repo_id})
        cur.commit()

    def repo_at(self, path: str) -> Repo | None:
        """The repo registered at *path*, matched exactly; `None` if there is none."""
        row = self.cursor().execute(_GET_REPO_SQL, {"path": path}).fetchone()
        return Repo(repo_id=int(row[0]), repo_path=path) if row else None

    def repos(self) -> list[Repo]:
        """Every registered repo, in the order they were registered."""
        rows = self.cursor().execute(_LIST_REPOS_SQL).fetchall()
        return [Repo(repo_id=int(r[0]), repo_path=str(r[1])) for r in rows]

    def _bootstrap(self) -> None:
        """Create the schema and check the embedding version, in one transaction."""
        cur = self.cursor()
        cur.begin()
        cur.execute(_SCHEMA_SQL)
        cur.execute(_SET_SCHEMA_VERSION_SQL, {"schema_version": str(SCHEMA_VERSION)})
        self._check_embedding_version(cur)
        cur.commit()

    @staticmethod
    def _check_embedding_version(cur: duckdb.DuckDBPyConnection) -> None:
        """Clear embeddings if the format version or model changed."""
        rows = cur.execute(_GET_EMBEDDING_META_SQL).fetchall()
        stored: dict[str, str] = {str(r[0]): str(r[1]) for r in rows}

        stored_version = stored.get("embedding_version", "")
        stored_model = stored.get("embedding_model", "")
        current_version = str(EMBEDDING_FORMAT_VERSION)
        current_model = config.embedding_model

        if stored_version == current_version and stored_model == current_model:
            return

        n = cur.execute(_COUNT_EMBEDDINGS_SQL).fetchone()
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
            cur.execute(_CLEAR_ALL_EMBEDDINGS_SQL)
        cur.execute(_SET_EMBEDDING_VERSION_SQL, {"embedding_version": current_version})
        cur.execute(_SET_EMBEDDING_MODEL_SQL, {"embedding_model": current_model})

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
            # No meta table: a fresh DB.  The writable bootstrap builds it;
            # a read would fail on the first missing table.
            if self._writable:
                return
            self._con.close()
            msg = f"No index at {self.db_path}; run `rbtr watch` first."
            raise IndexNotBuiltError(msg) from None
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
                f"({SCHEMA_VERSION}); run `rbtr watch` to rebuild it."
            )
            raise RbtrError(msg)
        log.warning("index_schema_changed", stored=stored or "none", current=SCHEMA_VERSION)
        if self._con.execute(_HAS_FTS_INDEX_SQL).fetchone():
            self._con.execute(_DROP_FTS_INDEX_SQL)
        self._con.execute(_WIPE_SCHEMA_SQL)

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """This thread's cursor on the database.

        Caches one cursor per thread to avoid the overhead of
        `connection.cursor()` on every operation (which triggers
        `getcwd` + `stat` syscalls in DuckDB).
        """
        cur = getattr(self._local, "cur", None)
        if cur is None or self._local.built_against is not self._con:
            # First use on this thread, or compaction published a new
            # connection: bind a fresh cursor to the current one. A read
            # already in progress keeps the cursor it captured (see
            # `Index.reader`), so a swap never tears it.
            cur = self._con.cursor()
            self._local.cur = cur
            self._local.built_against = self._con
        return cur

    def compact(self) -> None:
        """Rewrite the database into a fresh file, live.

        Deleting rows does not shrink a DuckDB file: the freed space is
        kept inside it for reuse.  Copying the live data into a fresh
        file and swapping it in is the only way to hand that space back
        to the operating system.  The copy (search index included) is
        opened as its own connection, atomically renamed over the
        original, and published (read-copy-update): reads still on the
        old connection finish against the old file and rebind on their
        next call, so none is cut off.  A failure leaves the original
        file and connection untouched; the index is simply not shrunk.
        """
        self.require_writable()
        if self.db_path is None:
            return  # in-memory: nothing on disk to compact
        db_path = Path(self.db_path)
        # Unique per compaction: the previous compaction's live connection
        # still identifies its database by its own temp path, so reusing
        # the name would collide on ATTACH.
        tmp_path = db_path.with_name(f"{db_path.name}.compact-{uuid4().hex}")
        try:
            self._write_compacted_copy(tmp_path)
        except (OSError, duckdb.Error) as exc:
            tmp_path.unlink(missing_ok=True)
            log.warning("gc_compaction_failed", error=str(exc))
            return
        # Open the compacted copy as its own connection (a distinct path,
        # so DuckDB's one-instance-per-path cache gives a fresh instance),
        # make it canonical with an atomic rename (the open file handle
        # follows the inode), then publish it. The old connection is left
        # to its readers and closes by refcount once they have rebound.
        new_con: duckdb.DuckDBPyConnection | None = None
        try:
            new_con = self._connect(str(tmp_path))
            os.replace(tmp_path, db_path)
        except (OSError, duckdb.Error) as exc:
            if new_con is not None:
                new_con.close()
            tmp_path.unlink(missing_ok=True)
            log.warning("gc_compaction_failed", error=str(exc))
            return
        self._adopt_connection(new_con)

    def _write_compacted_copy(self, tmp_path: Path) -> None:
        """Copy the live database into a fresh file at *tmp_path*.

        The statements below are written inline, not loaded from `.sql`
        files, deliberately. Two of them interpolate a value into the SQL
        text, and interpolation can only be reviewed for safety when the
        value and the statement it lands in are visible together — hiding
        the template in another file makes that worse. DuckDB forces the
        interpolation here: it parses an ATTACH path as a string literal
        and a database name as an identifier, and *neither* can be a
        bound parameter (`ATTACH $p` / `ATTACH ?` are parser errors),
        nor is there a Python-API equivalent — `duckdb.DuckDBPyConnection`
        has no `attach`/`detach`/`copy` method (only `execute`). So each
        value is escaped for the exact quoting context it sits in.
        """
        tmp_path.unlink(missing_ok=True)
        # Capture the cursor once: the ATTACH, COPY and DETACH must all
        # run on the same connection (as `reader` does).
        cur = self.cursor()

        # Source database name, needed to name it in the COPY below. This
        # is the engine's own identifier for the open file (e.g. `index`),
        # produced by DuckDB, never user-supplied text.
        row = cur.execute("SELECT current_database()").fetchone()
        source = str(row[0]) if row is not None else ""

        # ATTACH the (empty) target file. `tmp_path` derives from the
        # configured index path, not per-call input; it sits inside a
        # single-quoted string literal, so every single quote in it is
        # doubled ('->'') to keep the literal well-formed and unbreakable.
        attach_path = str(tmp_path).replace("'", "''")
        cur.execute(f"ATTACH '{attach_path}' AS compact_target")
        try:
            # Copy every table — chunk/snapshot/edge data and the FTS
            # index — into the target. `source` sits inside a
            # double-quoted identifier, so every double quote in it is
            # doubled ("->"") for the same reason.
            safe_source = source.replace('"', '""')
            cur.execute(f'COPY FROM DATABASE "{safe_source}" TO compact_target')
        finally:
            cur.execute("DETACH compact_target")

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
        row = self.cursor().execute(_DATA_SIZE_BYTES_SQL).fetchone()
        return int(row[0]) if row is not None else 0

    def disk_size_bytes(self) -> int:
        """Total bytes the index occupies on disk (database file + WAL).

        The honest footprint a user sees: it counts the actual files, so
        it is non-zero as soon as anything is committed and never depends
        on whether the write-ahead log has been consolidated into the
        main file. Zero for an in-memory index.
        """
        if self.db_path is None:
            return 0
        db = Path(self.db_path)
        total = db.stat().st_size if db.exists() else 0
        wal = db.with_name(f"{db.name}.wal")
        if wal.exists():
            total += wal.stat().st_size
        return total

    @staticmethod
    def _connect(dsn: str) -> duckdb.DuckDBPyConnection:
        """Open *dsn* with the configured memory limit and thread count.

        Every connection to an index opens through here: the store's own,
        and the one compaction adopts, which is a DuckDB instance of its own.
        """
        return duckdb.connect(
            dsn,
            config={
                "memory_limit": f"{int(config.duckdb_memory_limit)} bytes",
                "threads": config.duckdb_threads,
            },
        )

    def _load_fts(self, con: duckdb.DuckDBPyConnection) -> None:
        """Install and load the full-text-search extension on *con*."""
        con.install_extension("fts")
        con.load_extension("fts")

    def _adopt_connection(self, con: duckdb.DuckDBPyConnection) -> None:
        """Publish a freshly-opened connection as the live one, lock-free.

        Compaction opens the rewritten file as its own connection
        (`compact`) and hands it here. Assigning `self._con`
        is atomic under the GIL, so a reader sees either the old
        connection or the new one, never a torn state; each thread's
        cached cursor was built against the old connection, so `cursor`
        rebinds it on next use. The old connection is **not** closed here
        — it stays alive for readers still using it and closes itself by
        refcount once the last has rebound, so no read is ever cut off
        mid-query.
        """
        self._load_fts(con)
        self._con = con
