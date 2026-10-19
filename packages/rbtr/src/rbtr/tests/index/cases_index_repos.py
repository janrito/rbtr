"""Scenarios for `Index.register` and `repos`.

Repo registration is the entry point for every scenario in every
family; this file isolates cases that exercise the API's own
guarantees (one repo per path, idempotence, listing order).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RepoSequence:
    """Ordered `register` calls and the paths `repos()` lists after them.

    Duplicates in `calls` exercise idempotence.
    """

    calls: list[str] = field(default_factory=list)
    expected_paths: list[str] = field(default_factory=list)


def case_empty_has_no_repos() -> RepoSequence:
    return RepoSequence()


def case_single_register() -> RepoSequence:
    return RepoSequence(calls=["/r"], expected_paths=["/r"])


def case_same_path_twice_is_idempotent() -> RepoSequence:
    return RepoSequence(calls=["/r", "/r"], expected_paths=["/r"])


def case_distinct_paths_listed_in_registration_order() -> RepoSequence:
    return RepoSequence(calls=["/c", "/a", "/b"], expected_paths=["/c", "/a", "/b"])
