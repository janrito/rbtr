"""The index's files: one DuckDB instance holding the registry and every project's file.

Only `Index` uses `Store` (an import-linter contract keeps it
so).  ARCHITECTURE's "Index files" and "Concurrency model" describe how.
"""

from __future__ import annotations

import contextlib
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import duckdb

# duckdb requires this imported before any thread uses a connection.
import pyarrow  # noqa: F401
import structlog

from rbtr.config import config, index_dir_of, own_dir
from rbtr.domain.models import Repo, SnapshotRef
from rbtr.errors import (
    IndexLockedError,
    IndexNotBuiltError,
    IndexSchemaTooNewError,
    RbtrError,
    RepoNotFoundError,
)
from rbtr.git import common_dir
from rbtr.index import load_sql
from rbtr.index.constants import EMBEDDING_FORMAT_VERSION, SCHEMA_VERSION
from rbtr.index.db import build_fts

log = structlog.get_logger(__name__)

_REGISTRY_SQL = load_sql("registry.sql")
_SCHEMA_SQL = load_sql("schema.sql")
_GET_SCHEMA_VERSION_SQL = load_sql("get_schema_version.sql")
_SET_SCHEMA_VERSION_SQL = load_sql("set_schema_version.sql")
_LIST_PROJECTS_SQL = load_sql("list_projects.sql")
_LIST_REPOS_SQL = load_sql("list_repos.sql")
_GET_EMBEDDING_META_SQL = load_sql("get_embedding_meta.sql")
_COUNT_EMBEDDINGS_SQL = load_sql("count_embeddings.sql")
_CLEAR_ALL_EMBEDDINGS_SQL = load_sql("clear_all_embeddings.sql")
_SET_EMBEDDING_VERSION_SQL = load_sql("set_embedding_version.sql")
_SET_EMBEDDING_MODEL_SQL = load_sql("set_embedding_model.sql")
_GET_PROJECT_SQL = load_sql("get_project.sql")
_INSERT_PROJECT_SQL = load_sql("insert_project.sql")
_SET_PROJECT_FILE_SQL = load_sql("set_project_file.sql")
_DELETE_PROJECT_SQL = load_sql("delete_project.sql")
_REGISTER_REPO_SQL = load_sql("register_repo.sql")
_FORGET_REPO_SQL = load_sql("delete_repo.sql")
# A repo's rows in its project's file, deleted when it is forgotten.
_FORGET_ROWS_SQL = [
    load_sql("delete_snapshots_for_repo.sql"),
    load_sql("delete_edges_for_repo.sql"),
    load_sql("delete_indexed_snapshots_for_repo.sql"),
    load_sql("delete_watched_refs_for_repo.sql"),
]
_LIST_MISSING_REPOS_SQL = load_sql("list_missing_repos.sql")
_SET_MISSING_SINCE_SQL = load_sql("set_missing_since.sql")


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
    """One in-memory DuckDB instance with the index's files attached.

    The registry (`index.duckdb`) is attached as `registry`, and every
    project's data file (`index/<uuid>.duckdb`), which the registry
    names, under its alias.  Holding the registry is owning the index:
    DuckDB's file lock on it keeps every other process out, and every
    live data file is attached by the same process.  Every change to the
    registry is one of its methods, which reloads the copy that routes
    reads and writes.  Methods may be called from any thread: each thread
    gets its own cursor per file.
    """

    def __init__(self, db_path: Path | str | None = None, *, writable: bool = False) -> None:
        self.db_path: str | None = str(db_path) if db_path else None
        self._writable = writable
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(thread_name_prefix="rbtr-read")
        # A copy of the registry, reloaded by `sync`.
        self._project_of: dict[int, int] = {}
        self._file_of: dict[int, str] = {}
        self._repos: dict[str, Repo] = {}
        self._retired: list[tuple[str, float]] = []
        self._con = self._connect(":memory:")
        self._load_fts(self._con)
        if db_path is None:
            self._con.execute("ATTACH ':memory:' AS registry")
            self._create_registry()
            return
        path = Path(db_path)
        own_dir(path.parent)
        self._projects_dir.mkdir(exist_ok=True)
        # An in-memory instance spills to `.tmp` in the working directory
        # unless told otherwise; this is where a file database would.
        self._con.execute(f"SET temp_directory = {_quote(f'{path}.tmp')}")
        self._attach_registry()
        # Enforce the schema version *while holding the registry* (and
        # thus DuckDB's exclusive lock), so no second process can open
        # the index while it is checked or replaced.
        self._enforce_schema_version()
        self._sync()
        self._attach_projects()
        if writable:
            self._delete_unnamed_files()
            # Both schemas are `CREATE ... IF NOT EXISTS`: re-applying them
            # adds a table a newer rbtr introduced without a version bump.
            self._registry().execute(_REGISTRY_SQL)
            for file in self._file_of.values():
                cur = self._cursor_for(_alias(file))
                cur.execute(_SCHEMA_SQL)
                self._check_embedding_version(cur)

    def _attach_registry(self) -> None:
        """Attach `index.duckdb` as `registry`."""
        self._attach_file(str(self.db_path), "registry")

    def _attach_file(self, target: str, alias: str) -> None:
        """Attach *target* as *alias*, or close and raise `IndexLockedError`.

        DuckDB's lock error means another process holds the file, and
        with it the index; its file-handle conflict means another store in
        this process does.
        """
        try:
            self._con.execute(f"ATTACH {_quote(target)} AS {alias}")
        except (duckdb.IOException, duckdb.BinderException) as exc:
            message = str(exc).lower()
            if "lock" not in message and "file handle conflict" not in message:
                raise
            log.exception("duckdb_lock_conflict")
            self._con.close()
            msg = (
                "Index database is locked by another process. "
                "If the daemon is running, route commands through it "
                "(`rbtr daemon status` to check)."
            )
            raise IndexLockedError(msg) from exc

    def _create_registry(self) -> None:
        """Create the registry's tables and record this schema version."""
        cur = self._registry()
        cur.execute(_REGISTRY_SQL)
        cur.execute(_SET_SCHEMA_VERSION_SQL, {"schema_version": SCHEMA_VERSION})

    def _enforce_schema_version(self) -> None:
        """Keep, replace, or refuse the registry by its stored schema version.

        An equal version keeps the index; a newer running binary replaces
        it with an empty one (rbtr rebuilds rather than migrates); an
        *older* running binary refuses (raises `IndexSchemaTooNewError`)
        rather than destroy an index a newer rbtr wrote.  A fresh file
        gets the registry's tables when writable.
        """
        try:
            rows = self._registry().execute(_GET_SCHEMA_VERSION_SQL).fetchall()
        except duckdb.CatalogException:
            if self._writable:
                self._create_registry()
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
        # Deleting rows does not shrink a DuckDB file, so the old index is
        # replaced, not emptied.  Its path is unlinked while it is still
        # open, so its lock is held until a new file exists at the path:
        # a process that creates one first wins DuckDB's lock on it, and
        # the attach below then raises `IndexLockedError`.
        path = Path(str(self.db_path))
        path.unlink()
        Path(f"{path}.wal").unlink(missing_ok=True)
        self._con.execute("DETACH registry")
        self._local.cursors = {}
        self._attach_registry()
        self._create_registry()

    def _sync(self) -> None:
        """Reload the copy of the registry, and retire every file it stopped naming.

        Run after each registry transaction.  A file is attached before a
        registry row names it (`_new_file`, `_copy`), so this attaches
        nothing.
        """
        cur = self._registry()
        projects = cur.execute(_LIST_PROJECTS_SQL).fetchall()
        repos = cur.execute(_LIST_REPOS_SQL).fetchall()
        dropped = set(self._file_of.values()) - {str(r[1]) for r in projects}
        self._file_of = {int(r[0]): str(r[1]) for r in projects}
        self._project_of = {int(r[0]): int(r[2]) for r in repos}
        self._repos = {str(r[1]): Repo(repo_id=int(r[0]), repo_path=str(r[1])) for r in repos}
        for file in dropped:
            self._retire(file)

    def _attach_projects(self) -> None:
        """Attach every file the registry names.

        Another process holding one of them owns the index through a
        registry this process can no longer see, so this one backs out
        with `IndexLockedError`.
        """
        for file in self._file_of.values():
            self._attach(file)

    def _delete_unnamed_files(self) -> None:
        """Delete files no registry row names: the residue of a crash.

        Everything under `index/` that is not a live file or its WAL was
        being built or retired when the process stopped.  Copies left by
        the compaction that once renamed `index.duckdb.compact-*` over the
        index go too.
        """
        named = set(self._file_of.values())
        named |= {f"{file}.wal" for file in named}
        for entry in self._projects_dir.iterdir():
            if entry.name not in named:
                log.info("deleted_unnamed_index_file", file=entry.name)
                _delete(entry)
        path = Path(str(self.db_path))
        for entry in path.parent.glob(f"{path.name}.compact-*"):
            log.info("deleted_unnamed_index_file", file=entry.name)
            _delete(entry)

    @property
    def _projects_dir(self) -> Path:
        """The directory holding the project files, beside the registry."""
        return index_dir_of(Path(str(self.db_path)))

    @staticmethod
    def _connect(dsn: str) -> duckdb.DuckDBPyConnection:
        """Open *dsn* with the configured memory limit and thread count.

        The store's one connection opens through here; every file it
        uses is attached to it, so the limit covers them all.
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

    def _attach(self, file: str) -> None:
        """Attach the project file named *file* under its alias."""
        target = ":memory:" if self.db_path is None else str(self._projects_dir / file)
        self._attach_file(target, _alias(file))

    def _new_file(self) -> str:
        """Create, attach and return a fresh project file with the data schema.

        The registry does not name it yet: until a registry transaction
        does, a crash leaves it for `_delete_unnamed_files`.
        """
        file = f"{uuid4().hex}.duckdb"
        self._attach(file)
        cur = self._cursor_for(_alias(file))
        cur.execute(_SCHEMA_SQL)
        self._check_embedding_version(cur)
        # Built on the empty table, so a search never meets a file without one.
        build_fts(cur)
        return file

    def _file_bytes(self, file: str) -> int:
        """Bytes *file* and its WAL take on disk; zero in memory."""
        if self.db_path is None:
            return 0
        path = self._projects_dir / file
        wal = path.with_name(f"{file}.wal")
        return sum(p.stat().st_size for p in (path, wal) if p.exists())

    def register(self, path: str) -> Repo:
        """Register a repo path and return the repo.

        Idempotent — returns the existing repo if already registered.
        *path* must be canonical, since it is what identifies the repo:
        two spellings of one checkout would become two repos.  Pass it
        through `normalise_repo_path` first, as `build_index` does.

        The checkout joins the project of its git common directory, so a
        linked worktree shares its main checkout's file.  A path git does
        not know is a project of its own.  A new project's file is made
        before one registry transaction names it and registers the repo.
        """
        self.require_writable()
        existing = self.repo_at(path)
        if existing is not None:
            return existing
        shared = common_dir(path) or path
        cur = self._registry()
        row = cur.execute(_GET_PROJECT_SQL, {"common_dir": shared}).fetchone()
        if row is None:
            file = self._new_file()
            project = {
                "common_dir": shared,
                "file": file,
                "compacted_bytes": self._file_bytes(file),
            }
            cur.begin()
            project_id = int(cur.execute(_INSERT_PROJECT_SQL, project).fetchall()[0][0])
        else:
            cur.begin()
            project_id = int(row[0])
        params = {"path": path, "project_id": project_id}
        repo_id = int(cur.execute(_REGISTER_REPO_SQL, params).fetchall()[0][0])
        cur.commit()
        self._sync()
        return Repo(repo_id=repo_id, repo_path=path)

    def forget(self, repo: Repo) -> None:
        """Forget a registered repo, and its project with its last checkout.

        A project's last checkout takes the project with it: one registry
        transaction deletes both rows, and the project's file is retired.
        Otherwise the repo's `file_snapshots`, `edges`, `indexed_snapshots`
        and `watched_refs` are deleted from the project's file in one
        transaction, then its registry row.  Its chunks stay: the project's
        other checkouts share them, so reclaiming the ones it orphans is
        gc's job.
        """
        self.require_writable()
        repo_id = repo.repo_id
        project_id = self._project(repo)
        last = self._checkouts(project_id) == 1
        cur = self._registry()
        if last:
            cur.begin()
            cur.execute(_FORGET_REPO_SQL, {"repo_id": repo_id})
            cur.execute(_DELETE_PROJECT_SQL, {"project_id": project_id})
            cur.commit()
        else:
            data = self.data(repo)
            data.begin()
            for sql in _FORGET_ROWS_SQL:
                data.execute(sql, {"repo_id": repo_id})
            data.commit()
            cur.execute(_FORGET_REPO_SQL, {"repo_id": repo_id})
        self._sync()

    def compact(self, repo: Repo) -> None:
        """Rewrite *repo*'s project file into a fresh one, live.

        The file is the project's, so its other checkouts' data is
        rewritten with it.

        Deleting rows does not shrink a DuckDB file: the freed space is
        kept inside it for reuse.  Copying the live data into a fresh file
        is the only way to hand that space back.  The copy (search index
        included) is made and checkpointed, then one registry transaction
        names the new file.  Reads that started on the old file finish on
        it; it is retired.  A failure leaves the old file live and the
        index simply not shrunk.
        """
        self.require_writable()
        self._compact_project(self._project(repo))

    def compact_all(self) -> None:
        """Rewrite every project file into a fresh one, as `compact` does."""
        self.require_writable()
        for project_id in list(self._file_of):
            self._compact_project(project_id)

    def _compact_project(self, project_id: int) -> None:
        new = self._copy(project_id)
        if new is None:
            return
        params = {
            "project_id": project_id,
            "file": new,
            "compacted_bytes": self._file_bytes(new),
        }
        self._registry().execute(_SET_PROJECT_FILE_SQL, params)
        self._sync()

    def compact_grown(self, ratio: float) -> int:
        """Compact every project file grown past *ratio* times its baseline.

        The baseline is the file's size when it was last written whole
        (`projects.compacted_bytes`).  Returns how many files were
        compacted.
        """
        self.require_writable()
        rows = self._registry().execute(_LIST_PROJECTS_SQL).fetchall()
        grown = [int(r[0]) for r in rows if self._file_bytes(str(r[1])) > ratio * int(r[2])]
        for project_id in grown:
            self._compact_project(project_id)
        return len(grown)

    def missing_since(self) -> dict[Repo, datetime]:
        """When each repo whose path was found gone was first found so."""
        rows = self._registry().execute(_LIST_MISSING_REPOS_SQL).fetchall()
        # Stored as UTC without a zone: DuckDB needs `pytz` to return one.
        return {
            Repo(repo_id=int(r[0]), repo_path=str(r[1])): r[2].replace(tzinfo=UTC) for r in rows
        }

    def set_missing_since(self, repo: Repo, when: datetime | None) -> None:
        """Record when *repo*'s path was first found gone; `None` clears it."""
        self.require_writable()
        stored = when.astimezone(UTC).replace(tzinfo=None) if when is not None else None
        self._registry().execute(
            _SET_MISSING_SINCE_SQL, {"repo_id": repo.repo_id, "missing_since": stored}
        )

    def _retire(self, file: str) -> None:
        """Detach and delete *file* once reads that may be using it are done."""
        self._retired.append((file, time.monotonic() + config.retire_delay_s))

    def retire_due(self) -> None:
        """Detach and delete every retired file whose delay has passed.

        A read started before its file was replaced runs on the old file
        to the end; `retire_delay_s` is far longer than any read.
        """
        now = time.monotonic()
        due = [file for file, at in self._retired if at <= now]
        self._retired = [(file, at) for file, at in self._retired if at > now]
        for file in due:
            self._detach(file)

    def _detach(self, file: str) -> None:
        """Detach *file* and delete it with its WAL."""
        self._con.execute(f"DETACH {_alias(file)}")
        self._unlink(file)

    def _unlink(self, file: str) -> None:
        """Delete *file* and its WAL from disk, if they are there."""
        if self.db_path is not None:
            for name in (file, f"{file}.wal"):
                (self._projects_dir / name).unlink(missing_ok=True)

    def _check_embedding_version(self, cur: duckdb.DuckDBPyConnection) -> None:
        """Clear a file's embeddings if the format version or model changed."""
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

    def _cursor_for(self, alias: str) -> duckdb.DuckDBPyConnection:
        """This thread's cursor on the file attached as *alias*.

        Caches one cursor per thread and file to avoid the overhead of
        `connection.cursor()` on every operation (which triggers `getcwd`
        + `stat` syscalls in DuckDB).
        """
        cursors: dict[str, duckdb.DuckDBPyConnection] = self._local.__dict__.setdefault(
            "cursors", {}
        )
        cur = cursors.get(alias)
        if cur is None:
            cur = self._con.cursor()
            # The alias is one the store made (`registry` or `_alias`).
            cur.execute(f"USE {alias}")
            cursors[alias] = cur
        return cur

    def _registry(self) -> duckdb.DuckDBPyConnection:
        """This thread's cursor on the registry."""
        return self._cursor_for("registry")

    def _alias_of(self, repo: Repo) -> str:
        """The alias of the file holding *repo*'s data."""
        return _alias(self._file_of[self._project(repo)])

    def data(self, repo: Repo) -> duckdb.DuckDBPyConnection:
        """This thread's cursor on the file holding *repo*'s data."""
        return self._cursor_for(self._alias_of(repo))

    def across[T](
        self,
        within: list[SnapshotRef],
        read: Callable[[duckdb.DuckDBPyConnection, list[SnapshotRef]], T],
    ) -> list[T]:
        """Run *read* on each project file *within* spans; one result each.

        *read* gets that file's cursor and the refs it holds.  Several
        files are read at once, each on a pool thread with its own cursor:
        a statement has a fixed cost on every file, which reading them in
        turn would pay once per file.
        """
        groups: dict[str, list[SnapshotRef]] = {}
        for ref in within:
            groups.setdefault(self._alias_of(ref.repo), []).append(ref)
        if len(groups) <= 1:
            return [read(self._cursor_for(alias), refs) for alias, refs in groups.items()]
        return list(
            self._pool.map(lambda item: read(self._cursor_for(item[0]), item[1]), groups.items())
        )

    def each_file[T](self, read: Callable[[duckdb.DuckDBPyConnection], T]) -> list[T]:
        """Run *read* on every live project file at once; one result each."""
        aliases = [_alias(file) for file in self._file_of.values()]
        return list(self._pool.map(lambda alias: read(self._cursor_for(alias)), aliases))

    def close(self) -> None:
        """Delete retired files and close the database connection."""
        self._pool.shutdown()
        for file, _at in self._retired:
            self._detach(file)
        self._retired = []
        self._con.close()

    def disk_size_bytes(self) -> int:
        """Total bytes the index occupies on disk: its files and their WALs.

        The registry and every live project file.  A file a compaction
        replaced is not counted: it is deleted once `retire_delay_s` has
        passed.  It counts the actual files, so it is non-zero as soon as
        anything is committed and never depends on whether a write-ahead
        log has been consolidated.  Zero for an in-memory store.
        """
        if self.db_path is None:
            return 0
        registry = Path(self.db_path)
        files = [registry, registry.with_name(f"{registry.name}.wal")]
        for file in self._file_of.values():
            files += [self._projects_dir / file, self._projects_dir / f"{file}.wal"]
        return sum(p.stat().st_size for p in files if p.exists())

    def require_writable(self) -> None:
        """Raise `RuntimeError` unless this process may change the index."""
        if not self._writable:
            msg = "Writing to the index requires a writable Index (pass writable=True)"
            raise RuntimeError(msg)

    def _project(self, repo: Repo) -> int:
        """The project *repo* belongs to; `RepoNotFoundError` if none."""
        project_id = self._project_of.get(repo.repo_id)
        if project_id is None:
            msg = f"Repo not registered: {repo.repo_path}"
            raise RepoNotFoundError(msg)
        return project_id

    def _checkouts(self, project_id: int) -> int:
        """How many registered repos belong to *project_id*."""
        return sum(1 for p in self._project_of.values() if p == project_id)

    def repo_at(self, path: str) -> Repo | None:
        """The repo registered at *path*, matched exactly; `None` if there is none."""
        return self._repos.get(path)

    def repos(self) -> list[Repo]:
        """Every registered repo, in the order they were registered."""
        return sorted(self._repos.values(), key=lambda repo: repo.repo_id)

    def _copy(self, project_id: int) -> str | None:
        """Copy *project_id*'s file into a fresh one; the copy's name, or `None`.

        The copy (search index included) is checkpointed.  The registry
        does not name it yet: until a registry transaction does, a crash leaves it for
        `_delete_unnamed_files`.  A failure deletes the copy and leaves the
        old file live.
        """
        old = self._file_of[project_id]
        new = f"{uuid4().hex}.duckdb"
        try:
            self._attach(new)
            cur = self._cursor_for(_alias(new))
            # Both aliases are ones the store made (`_alias`).
            cur.execute(f"COPY FROM DATABASE {_alias(old)} TO {_alias(new)}")
            cur.execute("CHECKPOINT")
        except (OSError, duckdb.Error) as exc:
            log.warning("gc_compaction_failed", error=str(exc))
            # The attach itself may be what failed.
            with contextlib.suppress(duckdb.Error):
                self._con.execute(f"DETACH {_alias(new)}")
            self._unlink(new)
            return None
        return new


def _alias(file: str) -> str:
    """The name a project file is attached under: `p_` and its uuid."""
    return f"p_{file.removesuffix('.duckdb')}"


def _quote(text: str) -> str:
    """*text* as a SQL string literal.

    DuckDB parses an `ATTACH` path or a `SET` value as a literal and takes
    neither as a bound parameter, so the value is interpolated: every
    single quote in it is doubled to keep the literal unbreakable.
    """
    escaped = text.replace("'", "''")
    return f"'{escaped}'"


def _delete(path: Path) -> None:
    """Delete the file or directory at *path*."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
