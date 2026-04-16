"""Tests for the serialised build queue."""

from __future__ import annotations

from rbtr.daemon.build_queue import BuildQueue
from rbtr.daemon.messages import Notification
from rbtr.daemon.repos import RepoManager
from rbtr.index.store import IndexStore


def test_submit_enqueues() -> None:
    store = IndexStore()
    mgr = RepoManager(store)
    notifications: list[Notification] = []
    q = BuildQueue(mgr, notify=notifications.append)
    q.submit("/repo-a", ["HEAD"])
    # Job is in the queue (not started — no worker)
    assert not q._queue.empty()


def test_active_repo_none_before_build() -> None:
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    assert q.active_repo is None


def test_submit_multiple() -> None:
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    q.submit("/repo-a", ["HEAD"])
    q.submit("/repo-b", ["main"])
    q.submit("/repo-a", ["HEAD"])  # duplicate is fine — blob dedup handles it
    assert q._queue.qsize() == 3
