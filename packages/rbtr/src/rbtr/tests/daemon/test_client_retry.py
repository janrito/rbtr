"""Tests for DaemonClient retry on recv timeout.

Uses real ZMQ sockets throughout — no mocking of the transport
layer.  `time.sleep` is patched so the exponential backoff
doesn't slow the suite.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Generator
from pathlib import Path

import pytest
import zmq
from pytest_mock import MockerFixture

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import StatusRequest, StatusResponse
from rbtr.daemon.server import DaemonServer
from rbtr.daemon.status import write_status
from rbtr.errors import DaemonBusyError


@pytest.fixture
def silent_endpoint(runtime_dir: Path) -> Generator[Path]:
    """A real ZMQ ROUTER that receives but never replies.

    Simulates the real-world failure mode: the daemon process
    is alive (PID in the status file) but not responding in
    time.  ROUTER is used instead of REP because REP has its
    own send/recv state machine; ROUTER can silently absorb
    requests.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    endpoint = f"ipc://{runtime_dir / 'daemon.rpc'}"
    sock.bind(endpoint)
    write_status(
        runtime_dir,
        pid=os.getpid(),
        rpc=endpoint,
        pub=f"ipc://{runtime_dir / 'daemon.pub'}",
        version="test",
    )
    yield runtime_dir
    sock.setsockopt(zmq.LINGER, 0)
    sock.close()
    ctx.term()


def test_retry_succeeds_after_transient_timeout(
    running_server: DaemonServer,
    fake_repo: str,
    mocker: MockerFixture,
) -> None:
    """A slow first response triggers a retry that succeeds.

    The handler sleeps beyond the client's recv timeout on the
    first call, then responds immediately on the second.  The
    client reconnects and retries, receiving the fast response.
    """
    calls = 0

    async def slow_then_fast(_request: object) -> StatusResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.15)
        return StatusResponse()

    running_server.register("status", slow_then_fast)
    mock_sleep = mocker.patch("rbtr.daemon.client.time.sleep")

    with DaemonClient(
        running_server.runtime_dir,
        recv_timeout_ms=50,
        max_retries=3,
    ) as client:
        resp = client.send(StatusRequest(repo_path=fake_repo))

    assert isinstance(resp, StatusResponse)
    assert mock_sleep.call_count >= 1


@pytest.mark.parametrize(
    ("max_retries", "expected_sleeps"),
    [(1, 1), (0, 0)],
    ids=["one_retry", "no_retries"],
)
def test_all_retries_exhausted_raises(
    silent_endpoint: Path,
    fake_repo: str,
    mocker: MockerFixture,
    max_retries: int,
    expected_sleeps: int,
) -> None:
    """DaemonBusyError is raised after all retries are spent."""
    mock_sleep = mocker.patch("rbtr.daemon.client.time.sleep")

    with (
        DaemonClient(
            silent_endpoint,
            recv_timeout_ms=50,
            max_retries=max_retries,
        ) as client,
        pytest.raises(DaemonBusyError),
    ):
        client.send(StatusRequest(repo_path=fake_repo))

    assert mock_sleep.call_count == expected_sleeps
