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


class DaemonBusyError(RbtrError):
    """Raised when the daemon is alive but the request couldn't be served.

    The daemon process exists (pid file present, pid alive) but the
    request timed out or the socket roundtrip failed.  Distinct from
    "no daemon" (handled by `try_daemon` returning None) so the CLI
    can refuse to silently fall back to inline mode -- inline reads
    would contend for DuckDB's process-level WAL lock.
    """
