"""Sync ZMQ client for the rbtr daemon.

Connects to the daemon's REQ socket at the endpoint stored in
`daemon.json` under `config.runtime_dir`.  Sends typed
`Request` models, receives typed `Response` models — both
validated through pydantic `TypeAdapter`.

The client is synchronous (plain `zmq.Socket`, not async)
because the CLI is a short-lived process with no event loop.
Timeouts: 10 s receive, 5 s send.

Usage::

    with DaemonClient() as client:
        resp = client.send(StatusRequest(repo_path="/path"))
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import TypeGuard

import structlog
import zmq
from pydantic import BaseModel

from rbtr.config import config
from rbtr.daemon.messages import (
    ErrorResponse,
    Request,
    Response,
    ShutdownRequest,
    response_adapter,
)
from rbtr.daemon.pidfile import is_pid_alive
from rbtr.daemon.status import DaemonStatus, read_status, remove_status
from rbtr.errors import DaemonBusyError, RbtrError

log = structlog.get_logger(__name__)


def _status() -> DaemonStatus | None:
    """Read the daemon status file, or None if missing."""
    return read_status(config.runtime_dir)


def is_daemon_running() -> bool:
    """Check whether a daemon is currently running."""
    status = _status()
    if status is None:
        return False
    return is_pid_alive(status.pid)


def _daemon_ready(status: DaemonStatus | None) -> TypeGuard[DaemonStatus]:
    """True when *status* describes a live daemon process.

    The readiness test is deliberately "is **any** live daemon
    up?", not "is the daemon I spawned up?".  Concurrent callers
    that lose the start race must accept the winner's daemon
    rather than waiting for their own (doomed) `serve` to bind.
    """
    return status is not None and is_pid_alive(status.pid)


def start_daemon(*, allow_missing_plugins: bool = False) -> DaemonStatus:
    """Start the daemon and wait for it to become ready.

    Cleans up stale state before spawning: if a status file
    points to a PID that no longer exists, removes the status
    file and any orphaned socket files.  This prevents the
    next `daemon serve` from failing on `EADDRINUSE` after a
    hard-crashed daemon left sockets behind.

    Concurrency-safe: if a daemon is already running (a
    concurrent caller won the race), it is reused instead of
    spawning a second `serve`.  If our own spawn loses the race,
    we terminate it and return the winner.

    Returns the daemon status on success. Raises `RbtrError`
    if no daemon becomes ready within the timeout.
    """
    config.runtime_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # Stale cleanup: status file with dead PID -> remove status
    # + orphan sockets.  A live PID is left alone.
    # NOTE: PID recycling could cause a false positive from
    # is_pid_alive; low-probability for a per-user tool.
    status = read_status(config.runtime_dir)
    if status is not None and not is_pid_alive(status.pid):
        remove_status(config.runtime_dir)
        config.daemon_rpc.unlink(missing_ok=True)
        config.daemon_pub.unlink(missing_ok=True)
        status = None

    # A live daemon already exists (e.g. a concurrent caller won
    # the race): reuse it rather than spawning a second serve.
    if _daemon_ready(status):
        return status

    # Propagate the active dir overrides to the spawned child so
    # the daemon resolves the same data_path / log_path / cache_path
    # that the parent is using.  Without this the child re-parses
    # CLI and falls back to platformdirs defaults, which would
    # make the parent watch a different runtime_dir than the
    # daemon actually binds.
    cmd = [sys.executable, "-m", "rbtr"]
    for flag, value in (
        ("--data-dir", config.data_dir),
        ("--config-dir", config.config_dir),
        ("--log-dir", config.log_dir),
        ("--cache-dir", config.cache_dir),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    cmd.extend(["daemon", "serve"])
    if allow_missing_plugins:
        cmd.append("--allow-missing-plugins")

    # The daemon configures its own rotating JSON log sink
    # (`configure_logging(to_file=True)`), so we no longer redirect the
    # child's streams to the log file — that would pin the rotated
    # inode and interleave raw stderr with structured records.
    proc = subprocess.Popen(  # noqa: S603 - trusted args
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for any live daemon to appear (it writes its status
    # file after binding sockets).  If the daemon that comes up
    # isn't the one we spawned, a concurrent caller won the race:
    # terminate our redundant serve (it would otherwise die on
    # the DuckDB lock) and return the winner.
    #
    # If our spawn exits without a daemon appearing, it failed for
    # real (e.g. the index was built by a newer rbtr and this one
    # refuses).  Fail fast rather than wait the full 5 s -- but
    # allow a short grace first, since a spawn that *lost* the race
    # also exits, and the winner's status file may lag a beat.
    grace_after_exit = 5  # 100 ms ticks to let a racing winner appear
    for _ in range(50):  # 5 s at 100 ms intervals
        time.sleep(0.1)
        status = _status()
        if _daemon_ready(status):
            if status.pid != proc.pid and proc.poll() is None:
                proc.terminate()
            return status
        if proc.poll() is not None:
            if grace_after_exit <= 0:
                break
            grace_after_exit -= 1

    proc.terminate()
    msg = f"Daemon failed to start. Check {config.daemon_log} for the reason."
    raise RbtrError(msg)


def stop_daemon(*, timeout: float = 10.0) -> None:
    """Stop the running daemon gracefully.

    Sends a `ShutdownRequest` first. If the daemon does not exit
    within *timeout* seconds, falls back to SIGTERM.
    """
    status = _status()
    if status is None:
        return  # already stopped

    pid = status.pid
    runtime_dir = config.runtime_dir

    # Try graceful ZMQ shutdown first. Best-effort — the SIGTERM
    # path below is the real stop. Log at debug so failures are
    # visible without noise on the happy path.
    try:
        with DaemonClient(runtime_dir) as client:
            client.send_or_raise(ShutdownRequest())
    except Exception:  # noqa: BLE001 — best-effort shutdown; anything can fail
        log.debug("graceful_shutdown_failed", exc_info=True)

    # Wait for the process to exit
    for _ in range(int(timeout / 0.5)):
        time.sleep(0.5)
        if not is_pid_alive(pid):
            remove_status(runtime_dir)
            return

    # Escalate: SIGTERM
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        remove_status(runtime_dir)
        return

    for _ in range(6):  # 3 s at 0.5 s intervals
        time.sleep(0.5)
        if not is_pid_alive(pid):
            remove_status(runtime_dir)
            return

    msg = f"Daemon (PID {pid}) did not stop cleanly. Check {config.daemon_log} for details."
    raise RbtrError(msg)


class DaemonClient:
    """Sync ZMQ REQ client.

    Reads the RPC endpoint from the status file on first `send()`.
    A single client instance holds one REQ socket — calls are
    serialised (ZMQ REQ enforces strict send/recv alternation).
    """

    def __init__(
        self,
        runtime_dir: Path | None = None,
        *,
        recv_timeout_ms: int | None = None,
    ) -> None:
        self._runtime_dir = runtime_dir or config.runtime_dir
        self._recv_timeout_ms = (
            recv_timeout_ms if recv_timeout_ms is not None else config.daemon_recv_timeout_ms
        )
        self._ctx: zmq.Context[zmq.Socket[bytes]] | None = None
        self._sock: zmq.Socket[bytes] | None = None

    def __enter__(self) -> DaemonClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def _connect(self) -> zmq.Socket[bytes]:
        """Read status file and connect the socket."""
        status = read_status(self._runtime_dir)
        if status is None:
            msg = "Daemon not running (no status file)"
            raise DaemonBusyError(msg)
        self._ctx = zmq.Context()
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, 5_000)
        sock.connect(status.rpc)
        self._sock = sock
        return sock

    def send(self, request: Request) -> Response:
        """Send a request, return the typed response.

        Raises `DaemonBusyError` if the daemon is unreachable.
        """
        sock = self._sock or self._connect()

        try:
            sock.send(request.model_dump_json().encode())
            raw = sock.recv()
        except zmq.ZMQError as exc:
            raise DaemonBusyError from exc

        return response_adapter.validate_json(raw)

    def send_or_raise(self, request: Request) -> Response:
        """Like `send`, but raises `RbtrError` on `ErrorResponse`."""
        resp = self.send(request)
        if isinstance(resp, ErrorResponse):
            raise RbtrError(resp.message)
        return resp

    def send_or_raise_as[R: BaseModel](self, response_type: type[R], request: Request) -> R:
        """Send *request*, return a response narrowed to *response_type*.

        Raises `RbtrError` on `ErrorResponse` or on a daemon
        response whose type is anything other than
        *response_type*.  Saves callers from having to pattern-
        match the `Response` union after every request.
        """
        resp = self.send_or_raise(request)
        if not isinstance(resp, response_type):
            msg = f"expected {response_type.__name__} from daemon; got {type(resp).__name__}"
            raise RbtrError(msg)
        return resp

    def close(self) -> None:
        """Close the connection."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None


def try_daemon(request: Request) -> Response | None:
    """Try to send a request to the daemon. Return None if not running.

    Returns None *only* when there is no live daemon process
    (no status file, or the recorded PID isn't alive).  When the
    daemon's PID is alive but the request fails (busy worker,
    timeout, etc.), raises `DaemonBusyError` rather than silently
    falling back to inline mode -- inline fallback against a
    healthy daemon causes WAL-lock contention on the shared
    DuckDB file (DuckDB takes a process-level lock).

    Daemon protocol errors (e.g. `ErrorResponse`) are returned
    normally.
    """
    runtime_dir = config.runtime_dir
    status = read_status(runtime_dir)
    if status is None or not is_pid_alive(status.pid):
        remove_status(runtime_dir)
        return None
    try:
        with DaemonClient(runtime_dir) as client:
            return client.send(request)
    except DaemonBusyError as exc:
        msg = (
            f"daemon is running (pid {status.pid}) but did not respond: {exc}. "
            "It may be busy indexing; try `rbtr daemon status` or wait and retry."
        )
        raise DaemonBusyError(msg) from exc
