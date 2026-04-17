"""Scenarios for ``indexed_commits`` completion tracking.

Each case describes the sequence of ``mark_indexed`` calls the
fixture should replay, and the expected state of ``has_indexed``
and ``list_indexed_commits`` afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LifecycleScenario:
    """Declarative lifecycle-family test data."""

    repo_paths: list[str] = field(default_factory=lambda: ["/default"])
    # Ordered ``mark_indexed`` operations: (repo_id, commit_sha).
    marks: list[tuple[int, str]] = field(default_factory=list)
    # Small delay inserted between consecutive marks so indexed_at
    # strictly increases.  Expressed as zero-based indices after
    # which to sleep for 10 ms.
    sleep_after: list[int] = field(default_factory=list)

    # Expectations keyed by (repo_id, commit_sha).
    has_indexed: dict[tuple[int, str], bool] = field(default_factory=dict)
    # Expected commit_sha order from ``list_indexed_commits``, per repo.
    list_order: dict[int, list[str]] = field(default_factory=dict)


# ── Base cases ───────────────────────────────────────────────────────


def case_unindexed_sha_reports_false() -> LifecycleScenario:
    return LifecycleScenario(
        has_indexed={(1, "deadbeef"): False},
        list_order={1: []},
    )


def case_single_mark_then_has_indexed() -> LifecycleScenario:
    return LifecycleScenario(
        marks=[(1, "abc123")],
        has_indexed={(1, "abc123"): True, (1, "unrelated"): False},
        list_order={1: ["abc123"]},
    )


def case_mark_is_scoped_per_repo() -> LifecycleScenario:
    return LifecycleScenario(
        repo_paths=["/a", "/b"],
        marks=[(1, "sha1")],
        has_indexed={(1, "sha1"): True, (2, "sha1"): False},
    )


def case_mark_is_idempotent() -> LifecycleScenario:
    """Second mark of the same (repo, sha) does not create a new row."""
    return LifecycleScenario(
        marks=[(1, "sha1"), (1, "sha1")],
        has_indexed={(1, "sha1"): True},
        list_order={1: ["sha1"]},
    )


def case_list_returns_newest_first() -> LifecycleScenario:
    """Older mark, sleep, newer mark: list is [newer, older]."""
    return LifecycleScenario(
        marks=[(1, "older"), (1, "newer")],
        sleep_after=[0],  # sleep after the first mark
        has_indexed={(1, "older"): True, (1, "newer"): True},
        list_order={1: ["newer", "older"]},
    )
