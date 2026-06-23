"""Ref watcher — detects watched refs and worktrees that need indexing.

Two polling functions, both **read-only** (no writes to the
index store):

- `poll_watched` — watched-ref staleness.  For every ref in a
  repo's `watched_refs` table, resolves it to a SHA and returns
  a `WatchedTarget` when that SHA has no `indexed_commits` row.
  Symbolic names (`"HEAD"`, `"main"`) are re-resolved each poll
  so moving refs track their tip; a bare SHA resolves to itself
  (one-shot).  `HEAD` is just the default watched ref.
- `poll_worktree` — working-tree staleness.  Computes the
  current tree SHA via `worktree_tree_sha` and checks
  `has_indexed`.  Returns `DirtyWorktree` when the tree is
  dirty and the current tree SHA is not yet indexed.

No in-memory state: the store's `watched_refs` (intent) and
`indexed_commits` (completion) tables are the single source of
truth.  See ARCHITECTURE.md's Watched refs section.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from rbtr.errors import RbtrError
from rbtr.git import resolve_ref, worktree_tree_sha
from rbtr.index.store import IndexStore

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class WatchedTarget:
    """A watched ref whose resolved SHA is not in `indexed_commits`.

    `ref` is the symbolic watched name (e.g. `"HEAD"`, `"main"`);
    `sha` is its current resolution.
    """

    repo_path: str
    ref: str
    sha: str


@dataclass(frozen=True)
class DirtyWorktree:
    """A repo whose working tree has uncommitted changes not yet indexed.

    `tree_sha` is the git tree SHA representing the current
    working-tree state, computed by `worktree_tree_sha`.
    """

    repo_path: str
    repo_id: int
    tree_sha: str


def poll_watched(store: IndexStore) -> list[WatchedTarget]:
    """Return every watched ref whose resolved SHA is not yet indexed.

    Iterates registered repos (`store.list_repos`) and their
    `watched_refs`, resolving each symbolic name to a SHA with
    `git.resolve_ref`.  Unresolvable refs (deleted branch, missing
    repo path) are silently skipped.  A SHA already present in
    `indexed_commits` is skipped.  Targets are de-duplicated by
    `(repo_path, sha)`, so two refs pointing at the same commit
    (e.g. `"HEAD"` and `"main"`) yield a single build.
    """
    out: list[WatchedTarget] = []
    seen: set[tuple[str, str]] = set()
    for repo_id, path in store.list_repos():
        for ref in store.list_watched_refs(repo_id):
            try:
                sha = resolve_ref(path, ref)
            except RbtrError:
                continue
            if store.has_indexed(repo_id, sha):
                continue
            if (path, sha) in seen:
                continue
            seen.add((path, sha))
            out.append(WatchedTarget(repo_path=path, ref=ref, sha=sha))
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
