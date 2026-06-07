"""Cases for `_resolve_read_ref` routing.

Each case returns a `RefScenario` with the requested ref, repo
state, and expected symbolic result.  The test fixture resolves
symbolic names (`"HEAD_SHA"`, `"TREE_SHA"`, etc.) to real SHAs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RefScenario:
    """Declarative description of a `_resolve_read_ref` test case."""

    requested_ref: str | None
    # Whether to dirty the working tree before the test.
    dirty_worktree: bool = False
    # Whether to pre-index the worktree tree SHA.
    tree_sha_indexed: bool = False
    # Symbolic name for the expected return value.
    expected: str = "HEAD_SHA"


def case_none_with_dirty_indexed() -> RefScenario:
    """`None` + dirty tree + tree SHA indexed → tree SHA."""
    return RefScenario(
        requested_ref=None,
        dirty_worktree=True,
        tree_sha_indexed=True,
        expected="TREE_SHA",
    )


def case_none_clean_tree() -> RefScenario:
    """`None` + clean tree → HEAD SHA."""
    return RefScenario(
        requested_ref=None,
        expected="HEAD_SHA",
    )


def case_none_dirty_not_indexed() -> RefScenario:
    """`None` + dirty tree but NOT indexed → HEAD SHA."""
    return RefScenario(
        requested_ref=None,
        dirty_worktree=True,
        expected="HEAD_SHA",
    )


def case_head_explicit() -> RefScenario:
    """`"HEAD"` always resolves to committed state, even with worktree."""
    return RefScenario(
        requested_ref="HEAD",
        dirty_worktree=True,
        tree_sha_indexed=True,
        expected="HEAD_SHA",
    )


def case_branch_name() -> RefScenario:
    """Explicit branch name resolves to its SHA."""
    return RefScenario(
        requested_ref="feature",
        expected="FEATURE_SHA",
    )
