"""Tests for the ZMQ daemon server and sync client."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import anyio
import pytest

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    ErrorResponse,
    PingRequest,
    PingResponse,
    ShutdownRequest,
    ShutdownResponse,
)
from rbtr.daemon.server import DaemonServer


@pytest.fixture
def sock_dir() -> Path:
    """Short temp dir for IPC sockets (avoids AF_UNIX path limit)."""
    return Path(tempfile.mkdtemp(prefix="rbtr"))


@pytest.fixture
def running_server(sock_dir: Path) -> DaemonServer:
    """Start a server in a background thread, shut down after test."""
    server = DaemonServer(sock_dir)
    t = threading.Thread(target=lambda: anyio.run(server.serve), daemon=True)
    t.start()

    rpc_path = sock_dir / "daemon.rpc"
    for _ in range(100):
        if rpc_path.exists():
            break
        time.sleep(0.02)

    yield server  # type: ignore[misc]

    server.request_shutdown()
    t.join(timeout=3)


# ── Ping ─────────────────────────────────────────────────────────────


def test_ping(running_server: DaemonServer) -> None:
    client = DaemonClient(running_server.sock_dir)
    resp = client.send(PingRequest())
    assert isinstance(resp, PingResponse)
    assert resp.version == "0.1.0"
    assert resp.uptime >= 0
    client.close()


# ── Shutdown ─────────────────────────────────────────────────────────


def test_shutdown(sock_dir: Path) -> None:
    server = DaemonServer(sock_dir)
    t = threading.Thread(target=lambda: anyio.run(server.serve), daemon=True)
    t.start()

    rpc_path = sock_dir / "daemon.rpc"
    for _ in range(100):
        if rpc_path.exists():
            break
        time.sleep(0.02)

    client = DaemonClient(sock_dir)
    resp = client.send(ShutdownRequest())
    assert isinstance(resp, ShutdownResponse)
    client.close()

    t.join(timeout=3)
    assert not t.is_alive()


# ── Unknown kind ─────────────────────────────────────────────────────


def test_garbage_request(running_server: DaemonServer) -> None:
    """Raw garbage bytes return an ErrorResponse."""
    import zmq

    from rbtr.daemon.messages import response_adapter

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"ipc://{running_server.sock_dir / 'daemon.rpc'}")
    sock.send(b"not json")
    raw = sock.recv()
    resp = response_adapter.validate_json(raw)
    assert isinstance(resp, ErrorResponse)
    sock.close()
    ctx.term()


# ── Client connection refused ────────────────────────────────────────


def test_client_connection_refused(sock_dir: Path) -> None:
    """Client raises ConnectionError when no server is listening."""
    client = DaemonClient(sock_dir)
    with pytest.raises(ConnectionError):
        client.send(PingRequest())


# ── Multiple requests ────────────────────────────────────────────────


def test_multiple_requests(running_server: DaemonServer) -> None:
    client = DaemonClient(running_server.sock_dir)
    r1 = client.send(PingRequest())
    r2 = client.send(PingRequest())
    assert isinstance(r1, PingResponse)
    assert isinstance(r2, PingResponse)
    client.close()
