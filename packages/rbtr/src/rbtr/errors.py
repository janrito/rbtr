"""Base exceptions for rbtr."""


class RbtrError(Exception):
    """Top-level error raised by rbtr operations."""


class DaemonBusyError(RbtrError):
    """Raised when the daemon is alive but the request couldn't be served.

    The daemon process exists (pid file present, pid alive) but the
    request timed out or the socket roundtrip failed.  Distinct from
    "no daemon" (handled by `try_daemon` returning None) so the CLI
    can refuse to silently fall back to inline mode -- inline reads
    would contend for DuckDB's process-level WAL lock.
    """
