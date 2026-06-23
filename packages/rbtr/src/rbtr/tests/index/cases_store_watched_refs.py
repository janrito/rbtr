"""Scenarios for the `watched_refs` store API.

Mirrors `cases_store_repos.py`: declarative operation sequences
applied in one session, then the expected per-repo listing.
Exercises idempotent add, multi add + remove, empty no-ops, and
per-repo isolation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WatchedRefScenario:
    """Watched-ref operations and the expected final listing.

    `repos` maps a short key to the path to register. `ops` is an
    ordered list of `(repo_key, op, refs)` where `op` is `"add"` or
    `"remove"`. `expected` maps each repo key to the watched refs
    `list_watched_refs` should return after the ops are applied.
    """

    repos: dict[str, str]
    ops: list[tuple[str, str, list[str]]]
    expected: dict[str, list[str]]


def case_idempotent_add() -> WatchedRefScenario:
    return WatchedRefScenario(
        repos={"r": "/r"},
        ops=[("r", "add", ["a", "a"])],
        expected={"r": ["a"]},
    )


def case_add_more_then_remove() -> WatchedRefScenario:
    return WatchedRefScenario(
        repos={"r": "/r"},
        ops=[("r", "add", ["a", "b"]), ("r", "remove", ["a"])],
        expected={"r": ["b"]},
    )


def case_empty_ops_are_noops() -> WatchedRefScenario:
    return WatchedRefScenario(
        repos={"r": "/r"},
        ops=[("r", "add", ["b"]), ("r", "add", []), ("r", "remove", [])],
        expected={"r": ["b"]},
    )


def case_remove_absent_ref_is_noop() -> WatchedRefScenario:
    return WatchedRefScenario(
        repos={"r": "/r"},
        ops=[("r", "add", ["a"]), ("r", "remove", ["b"])],  # "b" was never watched
        expected={"r": ["a"]},
    )


def case_scoped_per_repo() -> WatchedRefScenario:
    return WatchedRefScenario(
        repos={"a": "/a", "b": "/b"},
        ops=[("a", "add", ["main"]), ("b", "add", ["HEAD"])],
        expected={"a": ["main"], "b": ["HEAD"]},
    )
