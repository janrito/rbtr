"""Behavioural tests for ``rbtr.daemon.watcher.poll``.

One behaviour: ``poll(store)`` returns the set of repos whose
current HEAD is not recorded in ``indexed_commits``.  Scenarios
live in ``case_watcher.py`` and feed both test functions below.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.daemon.watcher import StaleHead, poll
from rbtr.index.store import IndexStore


@parametrize_with_cases(
    "store, expected",
    cases="tests.daemon.case_watcher",
    has_tag="no_stale",
)
def test_poll_returns_no_stale_heads(
    store: IndexStore, expected: list[StaleHead]
) -> None:
    assert poll(store) == expected


@parametrize_with_cases(
    "store, expected",
    cases="tests.daemon.case_watcher",
    has_tag="stale",
)
def test_poll_reports_stale_heads(
    store: IndexStore, expected: list[StaleHead]
) -> None:
    assert poll(store) == expected
