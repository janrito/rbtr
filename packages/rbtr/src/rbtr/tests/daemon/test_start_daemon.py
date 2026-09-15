"""Unit test for the daemon liveness read.

`live_status` answers "is a daemon running here?" — a status
file that exists and names a process that is alive.  Tested
with real pids: the current process is alive, an unused high
pid is dead, matching `test_pidfile.py`, so `is_pid_alive` runs
for real rather than being mocked.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rbtr.daemon.client import live_status
from rbtr.daemon.status import write_status


@pytest.fixture
def live_daemon_dir(runtime_dir: Path) -> Path:
    """Runtime dir whose status file names this (alive) process."""
    write_status(
        runtime_dir,
        pid=os.getpid(),
        rpc="ipc:///tmp/test-rbtr.rpc",
        pub="ipc:///tmp/test-rbtr.pub",
        version="0.0.0",
    )
    return runtime_dir


@pytest.fixture
def dead_daemon_dir(runtime_dir: Path) -> Path:
    """Runtime dir whose status file names a pid that does not exist."""
    write_status(
        runtime_dir,
        pid=99999999,
        rpc="ipc:///tmp/test-rbtr.rpc",
        pub="ipc:///tmp/test-rbtr.pub",
        version="0.0.0",
    )
    return runtime_dir


def test_a_status_file_with_a_live_pid_is_a_running_daemon(live_daemon_dir: Path) -> None:
    status = live_status(live_daemon_dir)
    assert status is not None
    assert status.pid == os.getpid()


def test_a_status_file_with_a_dead_pid_is_not(dead_daemon_dir: Path) -> None:
    assert live_status(dead_daemon_dir) is None


def test_no_status_file_is_not(runtime_dir: Path) -> None:
    assert live_status(runtime_dir) is None
