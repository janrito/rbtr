"""Tests for the ZMQ daemon server and sync client.

Uses shared fixtures from conftest.py. Tests exercise the full
send → dispatch → respond path through the actual ZMQ socket.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import duckdb
import pytest
import structlog
import zmq
from pytest_cases import parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    DaemonConfigRequest,
    ErrorCode,
    ErrorResponse,
    GcAllRequest,
    GcEndedNotification,
    GcProgressNotification,
    GcStartedNotification,
    OkResponse,
    ShutdownRequest,
    StatusRequest,
    StatusResponse,
    notification_adapter,
    response_adapter,
)
from rbtr.daemon.server import Daemon
from rbtr.domain.models import GcMode
from rbtr.errors import DaemonBusyError, RbtrError
from rbtr.index.index import Index
from rbtr.tests.daemon.cases_server import MalformedMessage, StopScenario

from .conftest import serving


@parametrize_with_cases("scenario", cases=".cases_server")
def test_how_a_daemon_stops(
    scenario: StopScenario, log_output: structlog.testing.LogCapture
) -> None:
    """What stops a serving daemon, what `serve()` raises, and what it logs."""
    raised: list[BaseException] = []

    def serve() -> None:
        try:
            asyncio.run(scenario.server.serve())
        except BaseException as exc:  # noqa: BLE001 - CancelledError is a BaseException
            raised.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        assert scenario.server.wait_ready(), "daemon did not start within timeout"
        scenario.stop()
        thread.join(timeout=10)
        assert not thread.is_alive(), "the daemon did not stop"
    finally:
        if thread.is_alive():
            scenario.server.request_shutdown()
            thread.join(timeout=30)
            assert not thread.is_alive(), "the daemon did not stop"

    assert [str(exc) for exc in raised] == scenario.raised
    assert [e["event"] for e in log_output.entries if e["log_level"] == "error"] == scenario.logged


def test_multiple_requests(running_daemon: Daemon, fake_repo: str) -> None:
    with DaemonClient(running_daemon.runtime_dir) as client:
        r1 = client.send(StatusRequest(repo_path=fake_repo))
        r2 = client.send(StatusRequest(repo_path=fake_repo))
    assert r1.kind == "status"
    assert r2.kind == "status"


def test_gc_publishes_its_start_progress_and_end(running_daemon: Daemon) -> None:
    """A gc over every repo publishes one `gc_progress` per repo, between its start and end."""
    sub: zmq.Socket[bytes] = zmq.Context.instance().socket(zmq.SUB)
    sub.connect(running_daemon.pub_addr)
    sub.subscribe(b"")
    time.sleep(0.5)  # the subscription reaches the daemon
    with DaemonClient(running_daemon.runtime_dir) as client:
        client.send(GcAllRequest(mode=GcMode.WATCHED))
    published = []
    while sub.poll(2000):
        published.append(notification_adapter.validate_json(sub.recv()))
    sub.close()

    gc_kinds = (GcStartedNotification, GcProgressNotification, GcEndedNotification)
    assert [n for n in published if isinstance(n, gc_kinds)] == [
        GcStartedNotification(),
        GcProgressNotification(current=1, total=1),
        GcEndedNotification(),
    ]


def test_an_idle_daemon_runs_maintenance(
    runtime_dir: Path,
    seeded_index: Index,
    fake_repo: str,
    daemon_commit: str,
    stub_embedding_model: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon idle for `maintenance_idle_s` runs a pass, published as a gc.

    The pass records the registered checkout whose path is gone as missing,
    which starts its grace period, collects the live repo, and rebuilds the
    FTS index no write changed.
    """
    monkeypatch.setattr(config, "maintenance_idle_s", 0.0)
    gone = seeded_index.register(str(tmp_path / "gone"))
    live = seeded_index.repo(fake_repo)
    with seeded_index.reader(live) as cur:
        cur.execute("PRAGMA drop_fts_index('chunks')")
    server = Daemon(
        runtime_dir, index=seeded_index, idle_poll_interval=1.0, busy_poll_interval=60.0
    )
    sub: zmq.Socket[bytes] = zmq.Context.instance().socket(zmq.SUB)
    with serving(server):
        sub.connect(server.pub_addr)
        sub.subscribe(b"")
        published = []
        while sub.poll(3000):
            published.append(notification_adapter.validate_json(sub.recv()))
            if isinstance(published[-1], GcEndedNotification):
                break
    sub.close()

    gc_kinds = (GcStartedNotification, GcProgressNotification, GcEndedNotification)
    assert [n for n in published if isinstance(n, gc_kinds)] == [
        GcStartedNotification(),
        GcProgressNotification(current=1, total=1),
        GcEndedNotification(),
    ]
    assert gone in seeded_index.missing_since()
    assert len(seeded_index.fulltext_matches("load_config", within=[live.at(daemon_commit)])) > 0


@parametrize_with_cases("message", cases=".cases_server", prefix="message_")
def test_garbage_returns_error(
    message: MalformedMessage, running_daemon: Daemon, fake_repo: str
) -> None:
    message.client.send_multipart(message.frames)
    frames = message.client.recv_multipart()
    assert frames[:-1] == message.reply_prefix
    response = response_adapter.validate_json(frames[-1])
    assert isinstance(response, ErrorResponse)
    assert response.code is ErrorCode.INVALID_REQUEST
    message.client.send_multipart(
        [*message.reply_prefix, StatusRequest(repo_path=fake_repo).model_dump_json().encode()]
    )
    frames = message.client.recv_multipart()
    assert frames[:-1] == message.reply_prefix
    assert response_adapter.validate_json(frames[-1]).kind == "status"


def test_message_without_return_envelope_does_not_stop_service(
    running_daemon: Daemon, fake_repo: str, rpc_peer: zmq.Socket[bytes]
) -> None:
    rpc_peer.send(b"no-delimiter")
    rpc_peer.send_multipart(
        [b"return-hop", b"", StatusRequest(repo_path=fake_repo).model_dump_json().encode()]
    )
    frames = rpc_peer.recv_multipart()
    assert frames[:-1] == [b"return-hop", b""]
    assert response_adapter.validate_json(frames[-1]).kind == "status"


def test_malformed_argument_returns_field_feedback(running_daemon: Daemon) -> None:
    """A structurally-invalid argument is rejected with per-field detail.

    The error names the offending field and echoes the received value,
    so the caller can see how the argument was mis-shaped — general
    feedback, not a hand-picked failure mode.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"ipc://{running_daemon.runtime_dir / 'daemon.rpc'}")
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


def test_json_encoded_list_is_decoded_then_validated(running_daemon: Daemon) -> None:
    """A JSON-encoded list arg is decoded, then validated by pydantic.

    The unwrap only decodes; it does not type-check. A decoded list of
    the wrong element type is therefore rejected by normal validation
    with field-level feedback — no duplicated checking.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"ipc://{running_daemon.runtime_dir / 'daemon.rpc'}")
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
    running_daemon: Daemon, fake_repo: str, tmp_path: Path
) -> None:
    """A request with a non-git repo_path must not crash the daemon.

    Regression: `normalise_repo_path` raised `RbtrError` outside the
    dispatch error handler, so a client whose cwd was not a git repo
    took the whole (shared) daemon down mid-job.
    """
    not_git = tmp_path / "not_a_repo"
    not_git.mkdir()

    with DaemonClient(running_daemon.runtime_dir) as client:
        bad = client.send(StatusRequest(repo_path=str(not_git)))
        assert isinstance(bad, ErrorResponse)
        assert bad.code == ErrorCode.REPO_NOT_FOUND

        # Daemon is still alive and serving other repos.
        good = client.send(StatusRequest(repo_path=fake_repo))
    assert good.kind == "status"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ValueError, ErrorCode.INTERNAL),
        (duckdb.OutOfMemoryException, ErrorCode.OUT_OF_MEMORY),
    ],
)
def test_handler_exception_returns_error(
    running_daemon: Daemon, error: type[Exception], code: ErrorCode, mocker: MockerFixture
) -> None:
    mocker.patch("rbtr.daemon.handlers.get_version", side_effect=error("handler broke"))

    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(DaemonConfigRequest())
    assert isinstance(resp, ErrorResponse)
    assert resp.code == code
    assert "handler broke" in resp.message


def test_connection_refused(runtime_dir: Path) -> None:
    with DaemonClient(runtime_dir) as client, pytest.raises(DaemonBusyError):
        client.send(ShutdownRequest())


def test_send_or_raise_on_success(running_daemon: Daemon) -> None:
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send_or_raise(ShutdownRequest())
    assert isinstance(resp, OkResponse)


def test_send_or_raise_on_error(running_daemon: Daemon, mocker: MockerFixture) -> None:
    mocker.patch("rbtr.daemon.handlers.get_version", side_effect=ValueError("boom"))

    with (
        DaemonClient(running_daemon.runtime_dir) as client,
        pytest.raises(RbtrError, match="boom"),
    ):
        client.send_or_raise(DaemonConfigRequest())


def test_send_or_raise_as_narrows_response(running_daemon: Daemon) -> None:
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send_or_raise_as(OkResponse, ShutdownRequest())
    # mypy would reject a .foo on resp if narrowing didn't work.
    assert resp.kind == "ok"


def test_send_or_raise_as_rejects_mismatched_response(
    running_daemon: Daemon,
) -> None:
    with (
        DaemonClient(running_daemon.runtime_dir) as client,
        pytest.raises(RbtrError, match=r"expected StatusResponse.*got OkResponse"),
    ):
        client.send_or_raise_as(StatusResponse, ShutdownRequest())
