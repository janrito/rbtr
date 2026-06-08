"""Tests for the ZMQ daemon server and sync client.

Uses shared fixtures from conftest.py. Tests exercise the full
send → dispatch → respond path through the actual ZMQ socket.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

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
    StatusResponse,
    response_adapter,
)
from rbtr.daemon.server import DaemonServer
from rbtr.errors import DaemonBusyError, RbtrError

# ── Ping / shutdown ──────────────────────────────────────────────────


def test_shutdown(runtime_dir: Path) -> None:
    """Shutdown test runs its own server so it can assert the thread exits."""
    server = DaemonServer(runtime_dir, store=None, idle_poll_interval=60.0, busy_poll_interval=60.0)
    t = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    t.start()
    assert server.wait_ready(), "daemon did not start within timeout"

    with DaemonClient(runtime_dir) as client:
        resp = client.send(ShutdownRequest())
    assert isinstance(resp, OkResponse)
    t.join(timeout=3)
    assert not t.is_alive()


def test_multiple_requests(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        r1 = client.send(StatusRequest(path=fake_repo))
        r2 = client.send(StatusRequest(path=fake_repo))
    assert r1.kind == "status"
    assert r2.kind == "status"


# ── Error handling ───────────────────────────────────────────────────


def test_garbage_returns_error(running_server: DaemonServer) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"ipc://{running_server.runtime_dir / 'daemon.rpc'}")
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

    with DaemonClient(running_server.runtime_dir) as client:
        resp = client.send(ShutdownRequest())
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.INTERNAL
    assert "handler broke" in resp.message


# ── Client behaviour ────────────────────────────────────────────────


def test_connection_refused(runtime_dir: Path) -> None:
    with DaemonClient(runtime_dir) as client, pytest.raises(DaemonBusyError):
        client.send(ShutdownRequest())


def test_send_or_raise_on_success(running_server: DaemonServer) -> None:
    with DaemonClient(running_server.runtime_dir) as client:
        resp = client.send_or_raise(ShutdownRequest())
    assert isinstance(resp, OkResponse)


def test_send_or_raise_on_error(running_server: DaemonServer) -> None:
    def fail(_request: object) -> ErrorResponse:
        return ErrorResponse(code=ErrorCode.INTERNAL, message="boom")

    running_server.register("shutdown", fail)

    with (
        DaemonClient(running_server.runtime_dir) as client,
        pytest.raises(RbtrError, match="boom"),
    ):
        client.send_or_raise(ShutdownRequest())


def test_send_or_raise_as_narrows_response(running_server: DaemonServer) -> None:
    with DaemonClient(running_server.runtime_dir) as client:
        resp = client.send_or_raise_as(OkResponse, ShutdownRequest())
    # mypy would reject a .foo on resp if narrowing didn't work.
    assert resp.kind == "ok"


def test_send_or_raise_as_rejects_mismatched_response(
    running_server: DaemonServer,
) -> None:
    with (
        DaemonClient(running_server.runtime_dir) as client,
        pytest.raises(RbtrError, match=r"expected StatusResponse.*got OkResponse"),
    ):
        client.send_or_raise_as(StatusResponse, ShutdownRequest())
