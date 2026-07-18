"""Base exceptions for rbtr."""


class RbtrError(Exception):
    """Top-level error raised by rbtr operations.

    Subclasses may set `error_code` to a daemon `ErrorCode` value
    so that `_dispatch` maps them to the correct protocol error.
    The default is `"internal"`.
    """

    error_code: str = "internal"


class IndexNotBuiltError(RbtrError):
    """Raised when a search requires an FTS index that hasn't been built yet."""

    error_code: str = "index_not_built"

    def __init__(self, message: str = "FTS index not built. Run `rbtr index` first.") -> None:
        super().__init__(message)


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
    """Raised when the daemon is alive but the request couldn't be served.

    The daemon process exists (pid file present, pid alive) but the
    request timed out or the socket roundtrip failed.  Distinct from
    "no daemon" (handled by `try_daemon` returning None) so the CLI
    can refuse to silently fall back to inline mode -- inline reads
    would contend for DuckDB's process-level WAL lock.
    """
