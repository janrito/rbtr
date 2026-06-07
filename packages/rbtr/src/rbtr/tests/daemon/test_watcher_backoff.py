"""Tests for watcher poll-interval back-off.

The `DaemonServer._watcher_loop` uses two poll intervals:
one while no job is active, a longer one while a job is
running. The selector is isolated in `_next_poll_interval`
so this test doesn't need to drive the full loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.config import config
from rbtr.daemon.server import DaemonServer


@pytest.fixture
def server(runtime_dir: Path) -> DaemonServer:
    return DaemonServer(
        runtime_dir,
        store=None,
        idle_poll_interval=5.0,
        busy_poll_interval=30.0,
    )


def test_idle_interval_when_no_active_key(server: DaemonServer) -> None:
    """With no active job — idle."""
    assert server._active_key is None
    assert server._next_poll_interval() == 5.0


def test_busy_interval_when_active_key_set(server: DaemonServer) -> None:
    server._active_key = "/repo-a"
    assert server._next_poll_interval() == 30.0


def test_intervals_default_to_config_values(runtime_dir: Path) -> None:
    """With no kwargs, the server pulls both intervals from `config`."""
    server = DaemonServer(runtime_dir, store=None)
    assert server._idle_poll_interval == config.idle_poll_interval
    assert server._busy_poll_interval == config.busy_poll_interval
