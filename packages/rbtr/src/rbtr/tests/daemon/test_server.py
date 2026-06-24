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
from rbtr.index.store import IndexStore

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


@pytest.fixture
def warmup_started() -> threading.Event:
    """Set once the stubbed warmup begins running."""
    return threading.Event()


@pytest.fixture
def warmup_gate() -> threading.Event:
    """Set by the test to let the stubbed warmup finish."""
    return threading.Event()


@pytest.fixture
def warming_daemon(
    runtime_dir: Path,
    unindexed_store: IndexStore,
    warmup_started: threading.Event,
    warmup_gate: threading.Event,
) -> DaemonServer:
    """A daemon whose embedder warmup blocks until `warmup_gate` is set.

    Forces a warmup task to stay in-flight so a shutdown lands
    mid-warmup and exercises the cancel path.  The reranker is
    disabled so a single warmup task suffices.
    """
    server = DaemonServer(
        runtime_dir, store=unindexed_store, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    server._warmup = True
    server._reranker = None

    def blocking_warmup() -> None:
        warmup_started.set()
        warmup_gate.wait(timeout=10)

    assert server._embedder is not None
    server._embedder.warmup = blocking_warmup  # type: ignore[method-assign]  # stub blocks warmup mid-flight
    return server


def test_shutdown_during_warmup_does_not_raise(
    runtime_dir: Path,
    warming_daemon: DaemonServer,
    warmup_started: threading.Event,
    warmup_gate: threading.Event,
) -> None:
    """Cancelling in-flight warmup on shutdown must not escape `serve()`.

    Regression: warmup tasks were awaited under
    `contextlib.suppress(Exception)`, which does not catch the
    `CancelledError` (a `BaseException`) from `task.cancel()`, so a
    stop during warmup crashed the daemon with a non-zero exit.
    """
    errors: list[BaseException] = []

    def run() -> None:
        try:
            asyncio.run(warming_daemon.serve())
        except BaseException as exc:  # noqa: BLE001 - must capture CancelledError too
            errors.append(exc)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        assert warming_daemon.wait_ready(), "daemon did not start within timeout"
        assert warmup_started.wait(timeout=2), "warmup task did not start"
        with DaemonClient(runtime_dir) as client:
            client.send(ShutdownRequest())
    finally:
        warmup_gate.set()
    t.join(timeout=5)
    assert not t.is_alive()
    assert errors == [], f"serve() raised on shutdown: {errors!r}"


def test_multiple_requests(running_server_with_index: DaemonServer, fake_repo: str) -> None:
    with DaemonClient(running_server_with_index.runtime_dir) as client:
        r1 = client.send(StatusRequest(repo_path=fake_repo))
        r2 = client.send(StatusRequest(repo_path=fake_repo))
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


def test_malformed_argument_returns_field_feedback(running_server: DaemonServer) -> None:
    """A structurally-invalid argument is rejected with per-field detail.

    The error names the offending field and echoes the received value,
    so the caller can see how the argument was mis-shaped — general
    feedback, not a hand-picked failure mode.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"ipc://{running_server.runtime_dir / 'daemon.rpc'}")
    # file_paths must be a list of strings; element 123 is not a string.
    sock.send(b'{"kind":"read_symbol","repo_path":"/r","symbol":"x","file_paths":[123]}')
    raw = sock.recv()
    resp = response_adapter.validate_json(raw)
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.INVALID_REQUEST
    assert "file_paths" in resp.message
    assert "123" in resp.message
    sock.close()
    ctx.term()


def test_json_encoded_list_is_decoded_then_validated(running_server: DaemonServer) -> None:
    """A JSON-encoded list arg is decoded, then validated by pydantic.

    The unwrap only decodes; it does not type-check. A decoded list of
    the wrong element type is therefore rejected by normal validation
    with field-level feedback — no duplicated checking.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"ipc://{running_server.runtime_dir / 'daemon.rpc'}")
    # file_paths delivered as a JSON string encoding a list of ints.
    sock.send(b'{"kind":"read_symbol","repo_path":"/r","symbol":"x","file_paths":["[1, 2]"]}')
    raw = sock.recv()
    resp = response_adapter.validate_json(raw)
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.INVALID_REQUEST
    assert "file_paths" in resp.message
    sock.close()
    ctx.term()


def test_non_git_repo_path_returns_error_and_daemon_survives(
    running_server_with_index: DaemonServer, fake_repo: str, tmp_path: Path
) -> None:
    """A request with a non-git repo_path must not crash the daemon.

    Regression: `normalise_repo_path` raised `RbtrError` outside the
    dispatch error handler, so a client whose cwd was not a git repo
    took the whole (shared) daemon down mid-job.
    """
    not_git = tmp_path / "not_a_repo"
    not_git.mkdir()

    with DaemonClient(running_server_with_index.runtime_dir) as client:
        bad = client.send(StatusRequest(repo_path=str(not_git)))
        assert isinstance(bad, ErrorResponse)
        assert bad.code == ErrorCode.REPO_NOT_FOUND

        # Daemon is still alive and serving other repos.
        good = client.send(StatusRequest(repo_path=fake_repo))
    assert good.kind == "status"


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
