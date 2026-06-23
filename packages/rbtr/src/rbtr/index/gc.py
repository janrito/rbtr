"""Garbage collection for the code index.

One entry point, `run_gc`, executes every supported mode
(`GcMode`) over a single repo.  The caller supplies a git
`pygit2.Repository` for ref resolution and an `IndexStore` for
the actual writes.  Both daemon handlers and the inline CLI
fallback call this function; the schema of its inputs is the
single source of truth for what `rbtr gc` can do.

Modes
-----

WATCHED
    The **default**.  Keep HEAD, every local branch / tag / note,
    **and** every ref in the repo's `watched_refs` table resolved
    to a SHA (unresolvable refs skipped).  So a routine GC keeps
    everything a branch points at plus anything explicitly watched
    (e.g. a bare SHA); it only drops genuinely unreferenced commits.
WATCHED_ONLY
    Keep HEAD plus the resolved `watched_refs` only — drops
    unwatched branches and tags.  The opt-in way to reclaim every
    ref that is not on the watch list.
HEAD_ONLY
    Keep the repo's current HEAD; drop every other indexed commit.
KEEP
    Keep the union of HEAD and the caller-supplied refs; drop rest.
ORPHANS
    Do not drop any indexed commits.  Only sweep residue from
    crashed builds (`sweep_orphan_commits`, which also removes
    orphaned chunks).
"""

from __future__ import annotations

import structlog

from rbtr.daemon.messages import GcMode
from rbtr.errors import RbtrError
from rbtr.git import (
    head_sha,
    local_ref_shas,
    resolve_ref,
    resolve_refs_to_shas,
    worktree_tree_sha,
)
from rbtr.index.models import GcCounts
from rbtr.index.store import IndexStore

log = structlog.get_logger(__name__)

# Reference namespaces kept by `WATCHED` (the default mode).
_KEPT_REF_PREFIXES: tuple[str, ...] = (
    "refs/heads/",
    "refs/tags/",
    "refs/notes/",
)


def run_gc(
    store: IndexStore,
    repo_path: str,
    repo_id: int,
    *,
    mode: GcMode,
    refs: list[str],
    dry_run: bool,
) -> GcCounts:
    """Execute a garbage-collection operation.

    Returns accumulated `GcCounts` across every deletion performed.
    When *dry_run* is true, computes the counts without writing to
    the database.
    """
    if mode is GcMode.ORPHANS:
        return _run_orphans_only(store, repo_id, dry_run=dry_run)

    keep_set = _resolve_keep_set(repo_path, store, repo_id, mode=mode, refs=refs)
    drop_set = _resolve_drop_set(
        store,
        repo_path=repo_path,
        repo_id=repo_id,
        keep_set=keep_set,
    )

    if dry_run:
        return _dry_run_counts(store, repo_id, drop_set=drop_set)

    with store.session() as session:
        session.sweep()
        total = GcCounts()
        for sha in drop_set:
            total = total + session.drop_commit(repo_id, sha)
        total = total + session.cleanup(repo_id)
    return total


def _run_orphans_only(store: IndexStore, repo_id: int, *, dry_run: bool) -> GcCounts:
    if dry_run:
        # ORPHANS dry-run is conservative: report zero rather than
        # adding a read-only counterpart to sweep_orphan_commits.
        return GcCounts()
    with store.session() as session:
        session.sweep()
        total = session.cleanup(repo_id)
    return total


def _resolve_keep_set(
    repo_path: str,
    store: IndexStore,
    repo_id: int,
    *,
    mode: GcMode,
    refs: list[str],
) -> set[str]:
    """Return SHAs that must be preserved for this mode."""
    head = head_sha(repo_path)
    if mode is GcMode.HEAD_ONLY:
        return {head}
    if mode is GcMode.KEEP:
        return {head, *resolve_refs_to_shas(repo_path, refs)}
    if mode is GcMode.WATCHED_ONLY:
        return {head, *_watched_shas(store, repo_id, repo_path)}
    # WATCHED (default): all local refs plus the resolved watch set.
    return {
        head,
        *local_ref_shas(repo_path, _KEPT_REF_PREFIXES),
        *_watched_shas(store, repo_id, repo_path),
    }


def _watched_shas(store: IndexStore, repo_id: int, repo_path: str) -> set[str]:
    """Resolve the repo's watched refs to SHAs, skipping unresolvable ones."""
    out: set[str] = set()
    for ref in store.list_watched_refs(repo_id):
        try:
            out.add(resolve_ref(repo_path, ref))
        except RbtrError:
            # A watched ref that no longer resolves (deleted branch)
            # keeps nothing; its old commit becomes collectable.
            continue
    return out


def _resolve_drop_set(
    store: IndexStore,
    *,
    repo_path: str,
    repo_id: int,
    keep_set: set[str],
) -> set[str]:
    """Return indexed SHAs to drop: everything outside *keep_set*."""
    indexed = {sha for sha, _at in store.list_indexed_commits(repo_id)}
    # Protect the current worktree tree SHA from GC.  Tree SHAs
    # are never reachable from a git ref (they're tree objects,
    # not commits) so indexed - keep_set would include them.
    # The *current* tree SHA should be preserved; stale ones are
    # dropped by the server's post-build cleanup.
    current_wt = worktree_tree_sha(repo_path)
    if current_wt is not None:
        indexed.discard(current_wt)
    return indexed - keep_set


def _dry_run_counts(store: IndexStore, repo_id: int, *, drop_set: set[str]) -> GcCounts:
    """Compute counts without writing.

    Accurate for commits / snapshots / edges.  The `chunks` field
    is left at zero because computing it exactly would require
    simulating the drops and walking the blob-reference graph, and
    DuckDB does not allow nested transactions that could be rolled
    back.  Users who need precise chunk counts should run a real GC.
    """
    total = GcCounts(commits=len(drop_set))
    snapshots = 0
    edges = 0
    for sha in drop_set:
        snapshots += store.count_snapshots_for_commit(repo_id, sha)
        edges += store.count_edges_for_commit(repo_id, sha)
    return GcCounts(
        commits=total.commits,
        snapshots=snapshots,
        edges=edges,
        chunks=0,
    )
