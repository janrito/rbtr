"""Tests for the serialised build queue."""

from __future__ import annotations

import time

from rbtr.daemon.build_queue import BuildQueue
from rbtr.daemon.messages import ActiveJob, Notification
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
    q._active_ref_key = ("/repo-a", ("HEAD",), False)
    q.submit("/repo-a", ["HEAD"])
    assert len(q._queue) == 0


def test_snapshot_status_empty() -> None:
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    active, pending = q.snapshot_status()
    assert active is None
    assert pending == []


def test_snapshot_status_pending_only() -> None:
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    q.submit("/repo-a", ["HEAD"])
    q.submit("/repo-b", ["main", "dev"])
    active, pending = q.snapshot_status()
    assert active is None
    assert [(p.repo, p.refs) for p in pending] == [
        ("/repo-a", ["HEAD"]),
        ("/repo-b", ["main", "dev"]),
    ]


def test_snapshot_status_with_active_job() -> None:
    """When a build is running, ``active_job`` is populated with the
    current phase and progress; ``pending`` reflects whatever else
    is queued behind it."""
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    # Simulate the worker having popped a job and updated progress.
    q._active_job = ActiveJob(
        repo="/repo-a",
        ref="abc123",
        phase="embedding",
        current=2048,
        total=7522,
        elapsed_seconds=0.0,
    )
    q._started_at = time.monotonic() - 5.0  # job "started" 5s ago
    q.submit("/repo-b", ["HEAD"])
    active, pending = q.snapshot_status()
    assert active is not None
    assert active.repo == "/repo-a"
    assert active.ref == "abc123"
    assert active.phase == "embedding"
    assert active.current == 2048
    assert active.total == 7522
    assert active.elapsed_seconds >= 5.0
    assert [(p.repo, p.refs) for p in pending] == [("/repo-b", ["HEAD"])]


def test_snapshot_status_is_isolated_from_further_mutation() -> None:
    """The snapshot is a copy; later submissions don't retroactively
    appear in it."""
    store = IndexStore()
    mgr = RepoManager(store)
    q = BuildQueue(mgr, notify=lambda _: None)
    q.submit("/repo-a", ["HEAD"])
    _active, pending = q.snapshot_status()
    q.submit("/repo-b", ["HEAD"])
    assert len(pending) == 1
    # Second snapshot sees both.
    _active2, pending2 = q.snapshot_status()
    assert len(pending2) == 2
