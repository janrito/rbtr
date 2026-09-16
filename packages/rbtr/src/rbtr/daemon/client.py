"""Sync ZMQ client for the rbtr daemon.

Connects to the daemon's REQ socket at the endpoint stored in
`daemon.json` under `config.runtime_dir`.  Sends typed
`Request` models, receives typed `Response` models — both
validated through pydantic `TypeAdapter`.

The client is synchronous (plain `zmq.Socket`, not async)
because the CLI is a short-lived process with no event loop.
A send has 5 s to leave; a reply is waited for as long as
`DaemonClient.send` describes.

Usage::

    with DaemonClient() as client:
        resp = client.send(StatusRequest(repo_path="/path"))
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType

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
from rbtr.daemon.status import DaemonStatus, is_pid_alive, read_status, remove_status
from rbtr.errors import DaemonBusyError, RbtrError

log = structlog.get_logger(__name__)


def live_status(runtime_dir: Path) -> DaemonStatus | None:
    """The status file of a daemon that is running, if there is one.

    `None` covers both "no status file" and "a status file whose
    process is gone", because a caller can act on neither.  The
    question is deliberately "is **any** live daemon up?", not "is
    the daemon I spawned up?": a caller that loses a start race
    must accept the winner's daemon rather than wait for its own
    (doomed) `serve` to bind.
    """
    status = read_status(runtime_dir)
    if status is None or not is_pid_alive(status.pid):
        return None
    return status


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
    if the spawned daemon exits without one becoming ready, or
    does not bind within the backstop timeout.
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
    if status is not None:
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

    # Structured logs go to the rotating JSON `daemon.log` via
    # `configure_logging(to_file=True)` inside the child.  Stdout
    # and stderr go to a separate `daemon.stderr` file (truncated
    # each start) so native crashes, uncaught exceptions, and
    # C-library output are captured without corrupting the JSON
    # log or pinning the rotated inode.
    stderr_fh = config.daemon_stderr.open("w")
    proc = subprocess.Popen(  # noqa: S603 - trusted args
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=stderr_fh,
        stderr=stderr_fh,
        start_new_session=True,
    )
    stderr_fh.close()  # child inherited the fd; parent doesn't need it

    # Wait for any live daemon to appear (it writes its status
    # file after binding sockets).  If the daemon that comes up
    # isn't the one we spawned, a concurrent caller won the race:
    # terminate our redundant serve (it would otherwise die on
    # the DuckDB lock) and return the winner.
    #
    # Give up only when our spawned child actually exits (real
    # failure, or we lost the race and the winner's status file
    # lags -- hence the short grace).  The deadline is a backstop
    # against a child that wedges before binding, not the normal
    # path: a slow cold start under load still binds within it.
    grace_after_exit = 5  # 100 ms ticks to let a racing winner appear
    deadline = time.monotonic() + config.daemon_start_timeout
    while time.monotonic() < deadline:
        time.sleep(0.1)
        status = live_status(config.runtime_dir)
        if status is not None:
            if status.pid != proc.pid and proc.poll() is None:
                proc.terminate()
            return status
        if proc.poll() is not None:
            if grace_after_exit <= 0:
                msg = (
                    f"Daemon failed to start. "
                    f"Check {config.daemon_log} and {config.daemon_stderr} for the reason."
                )
                raise RbtrError(msg)
            grace_after_exit -= 1

    proc.terminate()
    msg = (
        f"Daemon did not become ready within {config.daemon_start_timeout:g}s. "
        f"Check {config.daemon_log} and {config.daemon_stderr} for the reason."
    )
    raise RbtrError(msg)


def stop_daemon(*, timeout: float = 10.0) -> None:
    """Stop the running daemon gracefully.

    Sends a `ShutdownRequest` first. If the daemon does not exit
    within *timeout* seconds, falls back to SIGTERM.
    """
    status = read_status(config.runtime_dir)
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
    except Exception:  # best-effort shutdown; anything can fail
        log.debug("graceful_shutdown_failed", exc_info=True)

    if _exits_within(pid, timeout):
        remove_status(runtime_dir)
        return

    # Escalate: SIGTERM
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        remove_status(runtime_dir)
        return

    if _exits_within(pid, 3.0):
        remove_status(runtime_dir)
        return

    msg = f"Daemon (PID {pid}) did not stop cleanly. Check {config.daemon_log} for details."
    raise RbtrError(msg)


def _exits_within(pid: int, timeout: float) -> bool:
    """Whether *pid* stops existing within *timeout* seconds."""
    for _ in range(int(timeout / 0.5)):
        time.sleep(0.5)
        if not is_pid_alive(pid):
            return True
    return False


class DaemonClient:
    """Sync ZMQ REQ client that waits for a busy daemon.

    Reads the RPC endpoint from the status file on first `send()`.
    *wait_budget_s* is how long a caller will wait for any one
    reply; *liveness_interval_s* is how often, while waiting, the
    daemon's process is checked for still being there.
    """

    def __init__(
        self,
        runtime_dir: Path | None = None,
        *,
        wait_budget_s: float | None = None,
        liveness_interval_s: float = 5.0,
    ) -> None:
        self._runtime_dir = runtime_dir or config.runtime_dir
        self._wait_budget_s = (
            wait_budget_s if wait_budget_s is not None else config.daemon_wait_budget_s
        )
        self._liveness_interval_s = liveness_interval_s
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
        """Read status file and connect a fresh REQ socket."""
        status = read_status(self._runtime_dir)
        if status is None:
            msg = "Daemon not running (no status file)"
            raise DaemonBusyError(msg)
        self._ctx = zmq.Context()
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int(self._liveness_interval_s * 1000))
        sock.setsockopt(zmq.SNDTIMEO, 5_000)
        sock.connect(status.rpc)
        self._sock = sock
        return sock

    def send(self, request: Request) -> Response:
        """Send a request and wait for the daemon's reply.

        The request is sent once.  A reply that has not arrived
        within `liveness_interval_s` is waited for again, until
        `wait_budget_s` is spent or the daemon's process is gone;
        either raises `DaemonBusyError`.

        Sending again would be wrong on both counts a slow reply
        allows.  The daemon is serving the request — a second copy
        makes it do the work twice and lands behind the first — or
        the daemon has died, and no copy of the request will be
        answered by a process that is not there.  Over IPC a reply
        cannot simply be lost, which is the case a re-send exists
        for.
        """
        sock = self._sock or self._connect()
        sock.send(request.model_dump_json().encode())
        deadline = time.monotonic() + self._wait_budget_s
        # Half the budget gone is the point worth saying out loud: a
        # slow reply is ordinary, one this far through the patience of
        # its caller is not.  Said once per request, not per check.
        report_at = time.monotonic() + self._wait_budget_s / 2

        while True:
            try:
                return response_adapter.validate_json(sock.recv())
            except zmq.ZMQError as exc:
                waited = self._wait_budget_s - (deadline - time.monotonic())
                if live_status(self._runtime_dir) is None:
                    msg = f"Daemon stopped while waiting for a reply ({waited:.0f}s)"
                    raise DaemonBusyError(msg) from exc
                if time.monotonic() >= deadline:
                    msg = f"Daemon did not reply within {self._wait_budget_s:g}s"
                    raise DaemonBusyError(msg) from exc
                if time.monotonic() >= report_at:
                    report_at = math.inf
                    log.warning(
                        "daemon_slow_reply",
                        kind=request.kind,
                        waited_s=round(waited, 1),
                        budget_s=self._wait_budget_s,
                    )

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
    status = live_status(runtime_dir)
    if status is None:
        remove_status(runtime_dir)  # drops a status file left by a dead daemon
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
