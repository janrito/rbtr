"""Tests for PID file management."""

from __future__ import annotations

import os
from pathlib import Path

from rbtr.daemon.pidfile import clean_pid, is_daemon_running, is_pid_alive, read_pid, write_pid


def test_write_and_read(tmp_path: Path) -> None:
    path = tmp_path / "daemon.pid"
    write_pid(path, 12345)
    assert read_pid(path) == 12345


def test_read_missing(tmp_path: Path) -> None:
    assert read_pid(tmp_path / "daemon.pid") is None


def test_is_pid_alive_current() -> None:
    assert is_pid_alive(os.getpid())


def test_is_pid_alive_dead() -> None:
    assert not is_pid_alive(99999999)


def test_stale_pid_detected(tmp_path: Path) -> None:
    path = tmp_path / "daemon.pid"
    write_pid(path, 99999999)
    assert not is_daemon_running(path)


def test_running_pid_detected(tmp_path: Path) -> None:
    path = tmp_path / "daemon.pid"
    write_pid(path, os.getpid())
    assert is_daemon_running(path)


def test_clean_pid(tmp_path: Path) -> None:
    path = tmp_path / "daemon.pid"
    write_pid(path, 12345)
    clean_pid(path)
    assert not path.exists()


def test_clean_pid_missing(tmp_path: Path) -> None:
    """clean_pid on a missing file does not raise."""
    clean_pid(tmp_path / "nonexistent.pid")
