"""Base exceptions and process exit codes for rbtr."""

from dataclasses import dataclass
from enum import IntEnum


class ExitCode(IntEnum):
    """Status an rbtr process exits with when it does not succeed."""

    FAILED = 1  # a command reported its own failure
    ERROR = 2  # an `RbtrError` or rejected arguments reached `main`
    INDEX_LOCKED = 3  # `daemon serve` found the index locked by another process


class RbtrError(Exception):
    """Top-level error raised by rbtr operations.

    Subclasses may set `error_code` to a daemon `ErrorCode` value
    so that `_dispatch` maps them to the correct protocol error.
    The default is `"internal"`.
    """

    error_code: str = "internal"


@dataclass(frozen=True, slots=True, kw_only=True)
class MissingSnapshot:
    """The snapshot a read needed, and the ref the caller named it by.

    The repo is named by its path, as clients name it; the values are
    plain because errors sit below the domain layer. Internal-only:
    raised by a handler and read by the server, it never crosses the RPC
    boundary.
    """

    repo_path: str
    snapshot_sha: str
    ref: str


class BuildStopped(Exception):
    """Raised inside a build that was asked to stop.

    Not an `RbtrError`: nothing went wrong, and no caller reports it.
    """


class IndexNotBuiltError(RbtrError):
    """Raised when what a read needs is not indexed.

    `missing` names the snapshot when one snapshot was missing; it is
    `None` when the index's tables are absent, which a build of any repo
    creates.
    """

    error_code: str = "index_not_built"

    def __init__(
        self,
        message: str,
        *,
        missing: MissingSnapshot | None = None,
    ) -> None:
        super().__init__(message)
        self.missing = missing


class RepoNotFoundError(RbtrError):
    """Raised when a path is not a git checkout, or the index has no row for its repo."""

    error_code: str = "repo_not_found"


class IndexSchemaTooNewError(RbtrError):
    """Raised when the index was written by a newer rbtr than this one.

    The index is never migrated, and an older rbtr must not wipe a
    newer index -- that triggers a destructive rebuild loop between
    installs sharing one data dir.  So the older binary refuses to
    open the DB instead of destroying it.
    """

    def __init__(self, *, stored: str, code: str) -> None:
        super().__init__(
            f"Index schema {stored} was written by a newer rbtr; this "
            f"install only supports {code}. Refusing to open rather than "
            "rebuild a newer index. Upgrade rbtr, or delete the index to "
            "rebuild it from scratch."
        )


class IndexLockedError(RbtrError):
    """Raised when another process holds DuckDB's write lock.

    DuckDB takes a process-level lock on the database file, so a
    caller that meets this cannot proceed by opening the file
    itself -- an inline build would contend for the same lock, and
    one that won it would hold the lock for the length of the
    build, keeping the daemon from starting.  Callers must report
    this and stop rather than fall back to inline mode.
    """

    error_code: str = "index_locked"


class MissingLanguagePluginsError(RbtrError):
    """Raised when the index holds languages the running rbtr can't load.

    The index records which languages it was built with (chunk
    `language`).  If the running process has no plugin loaded for one
    of them, a rebuild re-extracts those files as raw line-chunks,
    churning the content-addressed store and its embeddings.  So the
    build/daemon refuses rather than silently degrade -- usually a
    wrong or partially-installed environment.  `--allow-missing-plugins`
    overrides it.
    """

    def __init__(self, *, missing: set[str]) -> None:
        languages = ", ".join(sorted(missing))
        super().__init__(
            f"The index has chunks in languages this rbtr has not loaded: "
            f"{languages}. Rebuilding would re-extract them as raw chunks "
            f"and churn the index and its embeddings, so rbtr refuses to "
            f"run. Check `rbtr config` for loaded plugins and fix the "
            f"environment, or pass --allow-missing-plugins to override."
        )


class DaemonBusyError(RbtrError):
    """Raised when the local RPC client cannot obtain a reply.

    Reports discovery, wait-budget or process-liveness failure with
    `daemon_busy`. After sending, this error leaves execution ambiguous:
    the daemon may have performed the operation without returning a reply.
    """

    error_code: str = "daemon_busy"
