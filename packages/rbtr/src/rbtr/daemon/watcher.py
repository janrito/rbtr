"""Ref watcher — detects repos whose HEAD is not indexed.

Two polling functions, both **read-only** (no writes to the
index store):

- `poll` — commit staleness.  Returns `StaleHead` for repos
  whose current HEAD has no `indexed_commits` row.
- `poll_worktree` — working-tree staleness.  Computes the
  current tree SHA via `worktree_tree_sha` and checks
  `has_indexed`.  Returns `DirtyWorktree` when the tree is
  dirty and the current tree SHA is not yet indexed.

No in-memory state means no startup seeding, no register/unregister
API, and no drift between the watcher and the index: the store's
`indexed_commits` table is the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from rbtr.git import read_head, worktree_tree_sha
from rbtr.index.store import IndexStore

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class StaleHead:
    """A repo whose current HEAD is not recorded in `indexed_commits`."""

    repo_path: str
    new_ref: str


@dataclass(frozen=True)
class DirtyWorktree:
    """A repo whose working tree has uncommitted changes not yet indexed.

    `tree_sha` is the git tree SHA representing the current
    working-tree state, computed by `worktree_tree_sha`.
    """

    repo_path: str
    repo_id: int
    tree_sha: str


def poll(store: IndexStore) -> list[StaleHead]:
    """Return every repo whose current HEAD has not been fully indexed.

    Iterates over registered repos (`store.list_repos`), reads each
    one's current HEAD with `git.read_head`, and yields a `StaleHead`
    for any commit that has no `indexed_commits` row. Repos whose
    HEAD cannot be read (unborn, missing path, permission error) are
    silently skipped.
    """
    out: list[StaleHead] = []
    for repo_id, path in store.list_repos():
        current = read_head(path)
        if current is None:
            continue
        if store.has_indexed(repo_id, current):
            continue
        out.append(StaleHead(repo_path=path, new_ref=current))
    return out


def poll_worktree(store: IndexStore) -> list[DirtyWorktree]:
    """Return every repo whose working tree is dirty and not yet indexed.

    For each registered repo, computes the current tree SHA via
    `worktree_tree_sha`.  If the tree is dirty (tree SHA differs
    from HEAD's tree) and `has_indexed(repo_id, tree_sha)` is
    False, returns a `DirtyWorktree` so the worker can rebuild.

    Read-only — never writes to the store.  All writes are done
    by the job worker thread via `WriteSession`.
    """
    out: list[DirtyWorktree] = []
    for repo_id, path in store.list_repos():
        sha = worktree_tree_sha(path)
        if sha is None:
            continue
        if store.has_indexed(repo_id, sha):
            continue
        out.append(DirtyWorktree(repo_path=path, repo_id=repo_id, tree_sha=sha))
    return out
