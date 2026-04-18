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
    assert len(q._queue) == 1


def test_active_repo_none_before_build() -> None:
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    assert q.active_repo is None


def test_submit_distinct_entries_survive() -> None:
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    q.submit("/repo-a", ["HEAD"])
    q.submit("/repo-b", ["main"])
    assert len(q._queue) == 2


def test_submit_deduplicates_identical() -> None:
    """Submit-time dedupe. Blob-level dedup remains the last line of
    defence, but the queue no longer holds identical twins."""
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    q.submit("/repo-a", ["HEAD"])
    q.submit("/repo-a", ["HEAD"])
    assert len(q._queue) == 1


def test_submit_same_repo_different_ref_lists_are_distinct() -> None:
    """Dedupe key is `(repo, tuple(refs))` — not just the repo.

    Submissions with different ref lists represent different build
    intents and must all survive as separate queue entries.
    """
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    q.submit("/repo-a", ["HEAD"])
    q.submit("/repo-a", ["sha1"])
    q.submit("/repo-a", ["HEAD", "sha1"])
    assert len(q._queue) == 3


def test_submit_rejects_when_active() -> None:
    """An identical ``(repo, refs)`` submission is deduped against
    the currently-building job, not just the queue."""
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    # Simulate a build in progress — the worker normally sets these.
    q.active_repo = "/repo-a"
    q._active_ref_key = ("/repo-a", ("HEAD",))
    q.submit("/repo-a", ["HEAD"])
    assert len(q._queue) == 0
