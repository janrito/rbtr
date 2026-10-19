"""Ref watcher — detects watched refs and worktrees that need indexing.

Two polling functions, both **read-only** (no writes to the
index store):

- `poll_watched` — watched-ref staleness.  For every ref in a
  repo's `watched_refs` table, resolves it to a SHA and returns
  a `WatchedTarget` when that SHA has no `indexed_snapshots` row.
  Symbolic names (`"HEAD"`, `"main"`) are re-resolved each poll
  so moving refs track their tip; a bare SHA resolves to itself
  (one-shot).  `HEAD` is just the default watched ref.
- `poll_worktree` — working-tree staleness.  Computes the
  current tree SHA via `worktree_tree_sha` and checks
  `has_indexed`.  Returns `DirtyWorktree` when the tree is
  dirty and the current tree SHA is not yet indexed.

No in-memory state: the store's `watched_refs` (intent) and
`indexed_snapshots` (completion) tables are the single source of
truth.  See ARCHITECTURE.md's Watched refs section.
"""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass

import structlog

from rbtr.config import config
from rbtr.errors import RbtrError
from rbtr.git import resolve_ref, worktree_tree_sha
from rbtr.index.index import Index

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class WatchedTarget:
    """A watched ref whose resolved SHA is not in `indexed_snapshots`.

    `ref` is the symbolic watched name (e.g. `"HEAD"`, `"main"`);
    `snapshot_sha` is its current resolution.
    """

    repo_path: str
    ref: str
    snapshot_sha: str


@dataclass(frozen=True)
class DirtyWorktree:
    """A repo whose working tree has uncommitted changes not yet indexed.

    `snapshot_sha` is the git tree SHA representing the current
    working-tree state, computed by `worktree_tree_sha`.
    """

    repo_path: str
    repo_id: int
    snapshot_sha: str


@dataclass(frozen=True)
class FailedBuild:
    """A snapshot of a repo whose build raised."""

    repo_path: str
    snapshot_sha: str


def poll_watched(index: Index) -> list[WatchedTarget]:
    """Return every watched ref whose resolved SHA is not yet indexed.

    Iterates registered repos (`index.repos`) and their
    `watched_refs`, resolving each symbolic name to a SHA with
    `git.resolve_ref`.  Unresolvable refs (deleted branch, missing
    repo path) are silently skipped.  A SHA already present in
    `indexed_snapshots` is skipped.  Targets are de-duplicated by
    `(repo_path, sha)`, so two refs pointing at the same commit
    (e.g. `"HEAD"` and `"main"`) yield a single build.
    """
    out: list[WatchedTarget] = []
    seen: set[tuple[str, str]] = set()
    for repo in index.repos():
        for ref in index.watched_refs(repo):
            try:
                sha = resolve_ref(
                    repo.repo_path, ref, max_untracked_files=config.max_untracked_files
                )
            except RbtrError:
                continue
            if index.has_indexed(at=repo.at(sha)):
                continue
            if (repo.repo_path, sha) in seen:
                continue
            seen.add((repo.repo_path, sha))
            out.append(WatchedTarget(repo_path=repo.repo_path, ref=ref, snapshot_sha=sha))
    return out


def pending_builds(index: Index, failed: Set[FailedBuild]) -> list[WatchedTarget | DirtyWorktree]:
    """Return the builds that are due, watched refs first, skipping *failed*.

    The one place that answers "is there a build to run, and which".
    A watched ref outranks a dirty worktree, so the worktree scan is
    skipped entirely while any watched ref is stale — worth skipping,
    because `worktree_tree_sha` stats the whole tree and writes loose
    objects for a dirty one.
    """
    stale: list[WatchedTarget | DirtyWorktree] = [
        t for t in poll_watched(index) if FailedBuild(t.repo_path, t.snapshot_sha) not in failed
    ]
    if stale:
        return stale
    return [
        t for t in poll_worktree(index) if FailedBuild(t.repo_path, t.snapshot_sha) not in failed
    ]


def poll_worktree(index: Index) -> list[DirtyWorktree]:
    """Return every repo whose working tree is dirty and not yet indexed.

    For each registered repo, computes the current tree SHA via
    `worktree_tree_sha`.  If the tree is dirty (tree SHA differs
    from HEAD's tree) and `has_indexed` misses that tree, returns a
    `DirtyWorktree` so the worker can rebuild.

    Read-only — never writes to the store.  All writes are done
    by the job worker thread via `WriteSession`.
    """
    out: list[DirtyWorktree] = []
    for repo in index.repos():
        sha = worktree_tree_sha(repo.repo_path, max_untracked_files=config.max_untracked_files)
        if sha is None:
            continue
        if index.has_indexed(at=repo.at(sha)):
            continue
        out.append(DirtyWorktree(repo_path=repo.repo_path, repo_id=repo.repo_id, snapshot_sha=sha))
    return out
