"""Tests for the serialised build queue."""

from __future__ import annotations

from rbtr.daemon.build_queue import BuildQueue
from rbtr.daemon.repos import RepoManager
from rbtr.index.store import IndexStore


def _make_queue() -> BuildQueue:
    """BuildQueue with a real store but no worker started."""
    store = IndexStore()
    mgr = RepoManager(store)
    return BuildQueue(mgr, notify=lambda _n: None)


def test_submit_new_job() -> None:
    q = _make_queue()
    assert q.submit("/repo-a", ["HEAD"]) is True


def test_submit_duplicate_skipped() -> None:
    q = _make_queue()
    q.submit("/repo-a", ["HEAD"])
    assert q.submit("/repo-a", ["HEAD"]) is False


def test_submit_different_repos() -> None:
    q = _make_queue()
    assert q.submit("/repo-a", ["HEAD"]) is True
    assert q.submit("/repo-b", ["HEAD"]) is True


def test_is_busy_while_submitted() -> None:
    q = _make_queue()
    assert q.is_busy("/repo-a") is False
    q.submit("/repo-a", ["HEAD"])
    assert q.is_busy("/repo-a") is True


def test_not_busy_for_other_repo() -> None:
    q = _make_queue()
    q.submit("/repo-a", ["HEAD"])
    assert q.is_busy("/repo-b") is False
