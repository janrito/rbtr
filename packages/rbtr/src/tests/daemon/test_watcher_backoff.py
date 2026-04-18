"""Tests for watcher poll-interval back-off.

The ``DaemonServer._watcher_loop`` uses two poll intervals:
one while the build queue is idle, a longer one while a build
is active. The selector is isolated in ``_next_poll_interval``
so this test doesn't need to drive the full loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.daemon.build_queue import BuildQueue
from rbtr.daemon.repos import RepoManager
from rbtr.daemon.server import DaemonServer
from rbtr.index.store import IndexStore


@pytest.fixture
def server(sock_dir: Path) -> DaemonServer:
    return DaemonServer(
        sock_dir,
        store=None,
        idle_poll_interval=5.0,
        busy_poll_interval=30.0,
    )


@pytest.fixture
def build_queue() -> BuildQueue:
    return BuildQueue(RepoManager(IndexStore()), notify=lambda _: None)


def test_idle_interval_when_no_build_queue(server: DaemonServer) -> None:
    """With ``store=None`` there is no build queue at all — idle."""
    assert server._build_queue is None
    assert server._next_poll_interval() == 5.0


def test_idle_interval_when_queue_inactive(server: DaemonServer, build_queue: BuildQueue) -> None:
    server._build_queue = build_queue
    assert build_queue.active_repo is None
    assert server._next_poll_interval() == 5.0


def test_busy_interval_when_queue_active(server: DaemonServer, build_queue: BuildQueue) -> None:
    server._build_queue = build_queue
    build_queue.active_repo = "/repo-a"
    assert server._next_poll_interval() == 30.0
