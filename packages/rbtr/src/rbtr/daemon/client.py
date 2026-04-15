"""Sync ZMQ client for the rbtr daemon.

Connects to the daemon's REQ socket at
`ipc://<sock_dir>/daemon.rpc`. Sends typed `Request` models,
receives typed `Response` models — both validated through
pydantic `TypeAdapter`.

The client is synchronous (plain `zmq.Socket`, not async)
because the CLI is a short-lived process with no event loop.
Timeouts: 10 s receive, 5 s send.

Usage::

    client = DaemonClient(Path.home() / ".rbtr")
    resp = client.send(PingRequest())
    assert isinstance(resp, PingResponse)
    client.close()

For fire-and-forget from the CLI::

    resp = try_daemon(PingRequest())  # None if daemon not running
"""

from __future__ import annotations

from pathlib import Path

import zmq

from rbtr.config import config
from rbtr.daemon.messages import (
    ErrorResponse,
    Request,
    Response,
    response_adapter,
)


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

    def _connect(self) -> None:
        rpc_path = self._sock_dir / "daemon.rpc"
        if not rpc_path.exists():
            msg = f"Daemon not running (no socket at {rpc_path})"
            raise ConnectionError(msg)
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, 10_000)
        self._sock.setsockopt(zmq.SNDTIMEO, 5_000)
        self._sock.connect(self._rpc_addr)

    def send(self, request: Request) -> Response:
        """Send a request, return the typed response.

        Raises `ConnectionError` if the daemon is unreachable.
        """
        if self._sock is None:
            self._connect()

        sock = self._sock  # guaranteed non-None after _connect
        if sock is None:  # unreachable, but satisfies type checker
            msg = "Not connected"
            raise ConnectionError(msg)

        try:
            sock.send(request.model_dump_json().encode())
            raw = sock.recv()
        except zmq.ZMQError as exc:
            raise ConnectionError(f"Daemon communication failed: {exc}") from exc

        return response_adapter.validate_json(raw)

    def send_or_raise(self, request: Request) -> Response:
        """Like `send`, but raises on `ErrorResponse`."""
        resp = self.send(request)
        if isinstance(resp, ErrorResponse):
            raise Exception(resp.message)
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
    """Try to send a request to the daemon. Return None if not running."""
    sock_dir = Path(config.user_dir)
    rpc_path = sock_dir / "daemon.rpc"
    if not rpc_path.exists():
        return None
    client = DaemonClient(sock_dir)
    try:
        return client.send(request)
    except ConnectionError:
        rpc_path.unlink(missing_ok=True)
        return None
    finally:
        client.close()
