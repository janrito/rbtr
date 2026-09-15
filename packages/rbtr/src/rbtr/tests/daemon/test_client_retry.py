"""Tests for how `DaemonClient` waits when a reply is late.

Uses real ZMQ sockets throughout — no mocking of the transport
layer.  Budgets and the liveness interval are constructor
arguments, so a daemon "too slow to answer" is a handler that
sleeps for 150 ms.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path

import pytest
import zmq

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


@pytest.fixture
def dead_daemon_endpoint(silent_endpoint: Path) -> Path:
    """`silent_endpoint`, with the status file naming a dead PID.

    The PID is a subprocess that has already exited, so it is
    dead for certain rather than by assumption.
    """
    corpse = subprocess.Popen([sys.executable, "-c", ""])
    corpse.wait(timeout=30)
    write_status(
        silent_endpoint,
        pid=corpse.pid,
        rpc=f"ipc://{silent_endpoint / 'daemon.rpc'}",
        pub=f"ipc://{silent_endpoint / 'daemon.pub'}",
        version="test",
    )
    return silent_endpoint


def test_a_late_reply_is_waited_for_not_asked_for_again(
    running_daemon: DaemonServer,
    fake_repo: str,
) -> None:
    """A daemon slower than one liveness interval is waited for.

    The handler sleeps past the interval at which the client
    re-checks that the daemon is alive.  The client keeps
    waiting on the same socket, so the daemon serves the
    request once: re-sending would make a busy daemon do the
    same work twice over.
    """
    calls = 0

    async def slow(_request: object) -> StatusResponse:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.15)
        return StatusResponse()

    running_daemon._handlers["status"] = slow

    with DaemonClient(
        running_daemon.runtime_dir,
        wait_budget_s=5.0,
        liveness_interval_s=0.05,
    ) as client:
        resp = client.send(StatusRequest(repo_path=fake_repo))

    assert isinstance(resp, StatusResponse)
    assert calls == 1, f"the daemon served the request {calls} times"


def test_waiting_stops_when_the_budget_is_spent(
    silent_endpoint: Path,
    fake_repo: str,
) -> None:
    """A live daemon that never answers exhausts the budget."""
    with (
        DaemonClient(
            silent_endpoint,
            wait_budget_s=0.2,
            liveness_interval_s=0.05,
        ) as client,
        pytest.raises(DaemonBusyError),
    ):
        client.send(StatusRequest(repo_path=fake_repo))


def test_a_dead_daemon_is_not_waited_for(
    dead_daemon_endpoint: Path,
    fake_repo: str,
) -> None:
    """Waiting ends as soon as the daemon's process is gone.

    The endpoint still absorbs the request, so nothing but the
    PID says the daemon has died.  The client must notice on
    its first liveness check rather than spending the budget.
    """
    t0 = time.monotonic()
    with (
        DaemonClient(
            dead_daemon_endpoint,
            wait_budget_s=30.0,
            liveness_interval_s=0.05,
        ) as client,
        pytest.raises(DaemonBusyError),
    ):
        client.send(StatusRequest(repo_path=fake_repo))

    assert time.monotonic() - t0 < 1.0, "waited on a daemon that was not running"
