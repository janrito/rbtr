"""Sync ZMQ client for the rbtr daemon.

Connects to the daemon's REQ socket at the endpoint stored in
`daemon.json`. Sends typed `Request` models, receives typed
`Response` models — both validated through pydantic `TypeAdapter`.

The client is synchronous (plain `zmq.Socket`, not async)
because the CLI is a short-lived process with no event loop.
Timeouts: 10 s receive, 5 s send.

Usage::

    with DaemonClient() as client:
        resp = client.send(StatusRequest(repo="/path"))
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType

import zmq
from pydantic import BaseModel

from rbtr.config import config
from rbtr.daemon.messages import (
    ErrorResponse,
    Request,
    Response,
    response_adapter,
)
from rbtr.daemon.pidfile import is_pid_alive
from rbtr.daemon.status import DaemonStatus, read_status, remove_status
from rbtr.errors import DaemonBusyError, RbtrError

log = logging.getLogger(__name__)


def _sock_dir() -> Path:
    """Path to the daemon socket directory."""
    return config.home


def _status() -> DaemonStatus | None:
    """Read the daemon status file, or None if missing."""
    return read_status(_sock_dir())


def is_daemon_running() -> bool:
    """Check whether a daemon is currently running."""
    status = _status()
    if status is None:
        return False
    return is_pid_alive(status.pid)


def start_daemon() -> DaemonStatus:
    """Start the daemon and wait for it to become ready.

    Returns the daemon status on success. Raises `RuntimeError`
    if the daemon fails to start within the timeout.
    """
    home = _sock_dir()
    home.mkdir(parents=True, exist_ok=True)

    log_path = home / "daemon.log"
    # Explicit `--home` on the spawned child: otherwise the child
    # re-parses CLI without the parent's flag and falls back to
    # `~/.rbtr`, while the parent watches the wrong status file.
    with open(log_path, "a") as log:
        proc = subprocess.Popen(  # noqa: S603 - trusted args
            [sys.executable, "-m", "rbtr", "--home", str(home), "daemon", "serve"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Wait for the status file to appear (daemon writes it after bind)
    for _ in range(50):  # 5 s at 100 ms intervals
        time.sleep(0.1)
        status = _status()
        if status is not None and status.pid == proc.pid:
            return status

    proc.terminate()
    raise RuntimeError(f"Daemon failed to start within 5 s. Check {log_path} for details.")


def stop_daemon(*, timeout: float = 10.0) -> None:
    """Stop the running daemon gracefully.

    Sends a `ShutdownRequest` first. If the daemon does not exit
    within *timeout* seconds, falls back to SIGTERM.
    """
    status = _status()
    if status is None:
        return  # already stopped

    pid = status.pid

    # Try graceful ZMQ shutdown first. Best-effort — the SIGTERM
    # path below is the real stop. Log at debug so failures are
    # visible without noise on the happy path.
    try:
        with DaemonClient(_sock_dir()) as client:
            from rbtr.daemon.messages import ShutdownRequest

            client.send_or_raise(ShutdownRequest())
    except Exception:
        log.debug("Graceful ZMQ shutdown failed, falling back to signals", exc_info=True)

    # Wait for the process to exit
    for _ in range(int(timeout / 0.5)):
        time.sleep(0.5)
        if not is_pid_alive(pid):
            remove_status(_sock_dir())
            return

    # Escalate: SIGTERM
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        remove_status(_sock_dir())
        return

    for _ in range(6):  # 3 s at 0.5 s intervals
        time.sleep(0.5)
        if not is_pid_alive(pid):
            remove_status(_sock_dir())
            return

    raise RuntimeError(
        f"Daemon (PID {pid}) did not stop cleanly. Check {_sock_dir() / 'daemon.log'} for details."
    )


class DaemonClient:
    """Sync ZMQ REQ client.

    Reads the RPC endpoint from the status file on first `send()`.
    A single client instance holds one REQ socket — calls are
    serialised (ZMQ REQ enforces strict send/recv alternation).
    """

    def __init__(self, sock_dir: Path | None = None) -> None:
        self._sock_dir = sock_dir or _sock_dir()
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
        status = read_status(self._sock_dir)
        if status is None:
            msg = "Daemon not running (no status file)"
            raise ConnectionError(msg)
        self._ctx = zmq.Context()
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 10_000)
        sock.setsockopt(zmq.SNDTIMEO, 5_000)
        sock.connect(status.rpc)
        self._sock = sock
        return sock

    def send(self, request: Request) -> Response:
        """Send a request, return the typed response.

        Raises `ConnectionError` if the daemon is unreachable.
        """
        sock = self._sock or self._connect()

        try:
            sock.send(request.model_dump_json().encode())
            raw = sock.recv()
        except zmq.ZMQError as exc:
            raise ConnectionError(f"Daemon communication failed: {exc}") from exc

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
    sock_dir = _sock_dir()
    status = read_status(sock_dir)
    if status is None or not is_pid_alive(status.pid):
        remove_status(sock_dir)
        return None
    try:
        with DaemonClient(sock_dir) as client:
            return client.send(request)
    except ConnectionError as exc:
        msg = (
            f"daemon is running (pid {status.pid}) but did not respond: {exc}. "
            "It may be busy indexing; try `rbtr daemon status` or wait and retry."
        )
        raise DaemonBusyError(msg) from exc
