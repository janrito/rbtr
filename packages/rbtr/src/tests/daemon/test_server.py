"""Tests for the ZMQ daemon server and sync client.

Uses shared fixtures from conftest.py. Tests exercise the full
send → dispatch → respond path through the actual ZMQ socket.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import anyio
import pytest
import zmq

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    ErrorCode,
    ErrorResponse,
    OkResponse,
    Response,
    ShutdownRequest,
    StatusRequest,
    response_adapter,
)
from rbtr.daemon.server import DaemonServer
from rbtr.errors import RbtrError

# ── Ping / shutdown ──────────────────────────────────────────────────


def test_shutdown(sock_dir: Path) -> None:
    """Shutdown test runs its own server so it can assert the thread exits."""
    server = DaemonServer(sock_dir, store=None, poll_interval=60.0)
    t = threading.Thread(target=lambda: anyio.run(server.serve), daemon=True)
    t.start()
    rpc_path = sock_dir / "daemon.rpc"
    for _ in range(100):
        if rpc_path.exists():
            break
        time.sleep(0.02)

    with DaemonClient(sock_dir) as client:
        resp = client.send(ShutdownRequest())
    assert isinstance(resp, OkResponse)
    t.join(timeout=3)
    assert not t.is_alive()


def test_multiple_requests(running_server_with_index: DaemonServer) -> None:
    with DaemonClient(running_server_with_index.sock_dir) as client:
        r1 = client.send(StatusRequest(repo="/test/repo"))
        r2 = client.send(StatusRequest(repo="/test/repo"))
    assert r1.kind == "status"
    assert r2.kind == "status"


# ── Error handling ───────────────────────────────────────────────────


def test_garbage_returns_error(running_server: DaemonServer) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"ipc://{running_server.sock_dir / 'daemon.rpc'}")
    sock.send(b"not json")
    raw = sock.recv()
    resp = response_adapter.validate_json(raw)
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.INVALID_REQUEST
    sock.close()
    ctx.term()


def test_handler_exception_returns_error(running_server: DaemonServer) -> None:
    def bad_handler(_request: object) -> Response:
        msg = "handler broke"
        raise ValueError(msg)

    running_server.register("shutdown", bad_handler)

    with DaemonClient(running_server.sock_dir) as client:
        resp = client.send(ShutdownRequest())
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.INTERNAL
    assert "handler broke" in resp.message


# ── Client behaviour ────────────────────────────────────────────────


def test_connection_refused(sock_dir: Path) -> None:
    with DaemonClient(sock_dir) as client, pytest.raises(ConnectionError):
        client.send(ShutdownRequest())


def test_send_or_raise_on_success(running_server: DaemonServer) -> None:
    with DaemonClient(running_server.sock_dir) as client:
        resp = client.send_or_raise(ShutdownRequest())
    assert isinstance(resp, OkResponse)


def test_send_or_raise_on_error(running_server: DaemonServer) -> None:
    def fail(_request: object) -> ErrorResponse:
        return ErrorResponse(code=ErrorCode.INTERNAL, message="boom")

    running_server.register("shutdown", fail)

    with DaemonClient(running_server.sock_dir) as client, pytest.raises(RbtrError, match="boom"):
        client.send_or_raise(ShutdownRequest())
