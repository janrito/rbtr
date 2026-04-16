"""Sync ZMQ client for the rbtr daemon.

Connects to the daemon's REQ socket at
`ipc://<sock_dir>/daemon.rpc`. Sends typed `Request` models,
receives typed `Response` models — both validated through
pydantic `TypeAdapter`.

The client is synchronous (plain `zmq.Socket`, not async)
because the CLI is a short-lived process with no event loop.
Timeouts: 10 s receive, 5 s send.

Usage::

    with DaemonClient(Path.home() / ".rbtr") as client:
        resp = client.send(PingRequest())
        assert isinstance(resp, PingResponse)

For fire-and-forget from the CLI::

    resp = try_daemon(PingRequest())  # None if daemon not running
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import zmq

from rbtr.config import config
from rbtr.daemon.messages import (
    ErrorResponse,
    Request,
    Response,
    response_adapter,
)
from rbtr.errors import RbtrError


class DaemonClient:
    """Sync ZMQ REQ client.

    Connects lazily on first `send()`. A single client instance
    holds one REQ socket — calls are serialised (ZMQ REQ
    enforces strict send/recv alternation).
    """

    def __init__(self, sock_dir: Path) -> None:
        self._sock_dir = sock_dir
        self._rpc_addr = f"ipc://{sock_dir / 'daemon.rpc'}"
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
        """Connect and return the socket."""
        rpc_path = self._sock_dir / "daemon.rpc"
        if not rpc_path.exists():
            msg = f"Daemon not running (no socket at {rpc_path})"
            raise ConnectionError(msg)
        self._ctx = zmq.Context()
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 10_000)
        sock.setsockopt(zmq.SNDTIMEO, 5_000)
        sock.connect(self._rpc_addr)
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

    Only catches `ConnectionError` — daemon protocol errors
    (e.g. `ErrorResponse`) are returned normally.
    """
    sock_dir = Path(config.user_dir)
    rpc_path = sock_dir / "daemon.rpc"
    if not rpc_path.exists():
        return None
    with DaemonClient(sock_dir) as client:
        try:
            return client.send(request)
        except ConnectionError:
            rpc_path.unlink(missing_ok=True)
            return None
