"""Scenarios for ``IndexStore.register_repo`` and ``list_repos``.

Repo registration is the entry point for every scenario in every
family; this file isolates cases that exercise the API's own
guarantees (id issuance, idempotence, list roundtrip).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RepoSequence:
    """Ordered register_repo calls and expected ids.

    ``calls`` is a list of paths to register in order.  ``expected_ids``
    must have the same length and gives the id each call is expected
    to return.  Duplicates in ``calls`` exercise idempotence.
    """

    calls: list[str] = field(default_factory=list)
    expected_ids: list[int] = field(default_factory=list)
    # Expected state of list_repos() after all calls, as (id, path) tuples.
    expected_listing: list[tuple[int, str]] = field(default_factory=list)


def case_empty_has_no_repos() -> RepoSequence:
    return RepoSequence()


def case_single_register_returns_id_one() -> RepoSequence:
    return RepoSequence(
        calls=["/r"],
        expected_ids=[1],
        expected_listing=[(1, "/r")],
    )


def case_same_path_twice_is_idempotent() -> RepoSequence:
    return RepoSequence(
        calls=["/r", "/r"],
        expected_ids=[1, 1],
        expected_listing=[(1, "/r")],
    )


def case_distinct_paths_get_distinct_ids() -> RepoSequence:
    return RepoSequence(
        calls=["/a", "/b", "/c"],
        expected_ids=[1, 2, 3],
        expected_listing=[(1, "/a"), (2, "/b"), (3, "/c")],
    )
