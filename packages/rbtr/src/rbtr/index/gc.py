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
    read_head,
    resolve_ref,
    resolve_refs_to_shas,
    worktree_tree_sha,
)
from rbtr.index.models import GcCounts
from rbtr.index.store import IndexStore

log = structlog.get_logger(__name__)


def run_gc_all(
    store: IndexStore,
    *,
    mode: GcMode,
    refs: list[str],
    dry_run: bool,
) -> tuple[GcCounts, int]:
    """Run `run_gc` over every registered repo; sum counts, count repos.

    Backs `rbtr gc --all-repos`.  The chunk sweep is already global
    (content-addressed), so reclamation is correct regardless of iteration
    order; summing the per-repo `GcCounts` is exact (each chunk is freed in
    exactly the pass that drops its last reference). Returns the summed
    counts and the number of repos actually collected (skipped ones
    excluded).

    A repo whose path no longer resolves (a removed worktree/clone) is
    **skipped** with a hint, never purged — forgetting it is a separate,
    explicit action.
    """
    total = GcCounts()
    collected = 0
    for repo_id, repo_path in store.list_repos():
        if read_head(repo_path) is None:
            log.info("gc_skipped_unresolvable_repo", repo=repo_path)
            continue
        total = total + run_gc(store, repo_path, repo_id, mode=mode, refs=refs, dry_run=dry_run)
        collected += 1
    return total, collected


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
        # Predict only the chunks the drop set would free (read-only split,
        # graph intact). Pre-existing orphans the global prune would also
        # reclaim are added once by the caller (`handle_gc`).
        counts = _dry_run_counts(store, repo_id, drop_set=drop_set)
        chunks_freed, _kept = store.count_gc_chunk_split(repo_id, list(drop_set))
        return GcCounts(
            commits=counts.commits,
            snapshots=counts.snapshots,
            edges=counts.edges,
            chunks=chunks_freed,
        )

    # Real run: accumulate the *actual* deletions (commit drops + the
    # global orphan prune in `cleanup`), so the chunk figure is the true
    # reclamation — including orphans left by a forgotten repo.
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
    """Compute commit / snapshot / edge counts without writing.

    The chunk figures are computed separately by `run_gc` via
    `count_gc_chunk_split` (a read-only reference-graph query that
    needs no drop simulation), so this returns `chunks=0` and the
    caller overrides it.
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
