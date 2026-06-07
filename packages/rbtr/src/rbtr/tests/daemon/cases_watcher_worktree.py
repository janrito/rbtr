"""Scenarios for `rbtr.daemon.watcher.poll_worktree`.

Cases return a `WorktreeScenario` — pure declarative data
describing a repo's working-tree state and the expected
watcher output.  Shared fixtures in `test_watcher.py` convert
scenarios into real git repos and a real `IndexStore`.

Cases hold no I/O, no helpers, no references to pygit2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorktreeScenario:
    """Declarative description of a `poll_worktree` test scenario.

    `dirty_file` — if set, a tracked file is modified on disk
    after the initial commit.  `tree_sha_indexed` — whether
    the *current* worktree tree SHA has been pre-indexed.
    `expected_dirty` — whether `poll_worktree` should return
    a `DirtyWorktree` for this repo.
    """

    dirty_file: bool = False
    tree_sha_indexed: bool = False
    repo_exists: bool = True
    expected_dirty: bool = False


def case_clean_worktree() -> WorktreeScenario:
    """Clean tree → nothing happens (worktree_tree_sha returns None)."""
    return WorktreeScenario()


def case_dirty_not_indexed() -> WorktreeScenario:
    """Dirty tree, not yet indexed → returns DirtyWorktree."""
    return WorktreeScenario(dirty_file=True, expected_dirty=True)


def case_dirty_already_indexed() -> WorktreeScenario:
    """Dirty tree but tree SHA already indexed → skip (no infinite loop)."""
    return WorktreeScenario(dirty_file=True, tree_sha_indexed=True)


def case_missing_worktree_repo() -> WorktreeScenario:
    """Registered path gone → silently skipped."""
    return WorktreeScenario(repo_exists=False)
