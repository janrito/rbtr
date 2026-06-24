"""Unit test for the daemon readiness predicate.

`_daemon_ready` is pure: a status is "ready" when it exists and
its pid is alive.  Tested with real pids — the current process
is alive, an unused high pid is dead — matching `test_pidfile.py`,
so `is_pid_alive` runs for real rather than being mocked.
"""

from __future__ import annotations

import os

import pytest

from rbtr.daemon.client import _daemon_ready
from rbtr.daemon.status import DaemonStatus


@pytest.fixture
def live_status() -> DaemonStatus:
    """Status pointing at this (alive) process."""
    return DaemonStatus(
        pid=os.getpid(),
        rpc="ipc:///tmp/test-rbtr.rpc",
        pub="ipc:///tmp/test-rbtr.pub",
        started_at="2026-01-01T00:00:00Z",
        version="0.0.0",
    )


@pytest.fixture
def dead_status() -> DaemonStatus:
    """Status pointing at a pid that does not exist."""
    return DaemonStatus(
        pid=99999999,
        rpc="ipc:///tmp/test-rbtr.rpc",
        pub="ipc:///tmp/test-rbtr.pub",
        started_at="2026-01-01T00:00:00Z",
        version="0.0.0",
    )


def test_ready_when_status_pid_is_alive(live_status: DaemonStatus) -> None:
    assert _daemon_ready(live_status) is True


def test_not_ready_when_status_pid_is_dead(dead_status: DaemonStatus) -> None:
    assert _daemon_ready(dead_status) is False


def test_not_ready_when_no_status() -> None:
    assert _daemon_ready(None) is False
