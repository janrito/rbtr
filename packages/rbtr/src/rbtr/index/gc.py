"""Garbage collection for the code index.

One entry point, `run_gc`, executes every supported mode
(`GcMode`) over a single repo.  The caller supplies a git
`pygit2.Repository` for ref resolution and an `Index` for
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

from rbtr.config import config
from rbtr.domain.models import EveryRepoGcMode, GcCounts, GcMode, Repo
from rbtr.errors import RbtrError
from rbtr.git import (
    head_sha,
    local_ref_shas,
    read_head,
    resolve_ref,
    resolve_refs_to_shas,
    worktree_tree_sha,
)
from rbtr.index.index import Index
from rbtr.index.progress import RepoProgressCallback, _noop_repo_progress

log = structlog.get_logger(__name__)


def run_gc_all(
    index: Index,
    *,
    mode: EveryRepoGcMode,
    dry_run: bool,
    compact: bool = False,
    on_progress: RepoProgressCallback = _noop_repo_progress,
) -> tuple[GcCounts, int]:
    """Run `run_gc` over every registered repo; sum counts, count repos.

    Backs `rbtr gc --scope all`.  The chunk sweep is already global
    (content-addressed), so reclamation is correct regardless of iteration
    order; summing the per-repo `GcCounts` is exact (each chunk is freed in
    exactly the pass that drops its last reference). Returns the summed
    counts and the number of repos actually collected (skipped ones
    excluded).

    `on_progress` is called with each repo as it is collected, so a pass
    over many repos can be watched rather than waited on.

    A repo whose path no longer resolves (a removed worktree/clone) is
    **skipped** with a hint, never purged — forgetting it is a separate,
    explicit action.  A repo whose HEAD is unborn reads the same way and
    is skipped too: `run_gc` raises on one, and an empty checkout must
    not end a pass over every other repo.
    """
    live = [repo for repo in index.repos() if read_head(repo.repo_path) is not None]
    for repo in index.repos():
        if read_head(repo.repo_path) is None:
            log.info("gc_skipped_unresolvable_repo", repo=repo.repo_path)
    total = GcCounts()
    for collected, repo in enumerate(live, start=1):
        # Never compact per repo: a rewrite is expensive, so compact once
        # after every repo's logical deletes have landed.
        total = total + run_gc(index, repo.repo_path, mode=mode, refs=[], dry_run=dry_run)
        on_progress(repo.repo_path, collected, len(live))
    if compact and not dry_run:
        with index.session() as session:
            session.compact()
    return total, len(live)


# Reference namespaces kept by `WATCHED` (the default mode).
_KEPT_REF_PREFIXES: tuple[str, ...] = (
    "refs/heads/",
    "refs/tags/",
    "refs/notes/",
)


def run_gc(
    index: Index,
    repo_path: str,
    *,
    mode: GcMode,
    refs: list[str],
    dry_run: bool,
    compact: bool = False,
) -> GcCounts:
    """Execute a garbage-collection operation.

    Returns accumulated `GcCounts` across every deletion performed.
    When *dry_run* is true, computes the counts without writing to
    the database.  When *compact* is true, the index file is rewritten
    after the deletions commit, reclaiming the freed disk space.

    *repo_path* must already be registered — collecting a repo that was
    never indexed is meaningless, so this raises rather than creating
    one.  The repo is looked up from it, so the two cannot disagree.
    """
    repo = index.repo(repo_path)
    if mode is GcMode.ORPHANS:
        return _run_orphans_only(index, repo, dry_run=dry_run, compact=compact)

    keep_set = _resolve_keep_set(index, repo, mode=mode, refs=refs)
    drop_set = _resolve_drop_set(index, repo, keep_set=keep_set)

    if dry_run:
        # Predict only the chunks the drop set would free (read-only split,
        # graph intact). Pre-existing orphans the global prune would also
        # reclaim are added once by the caller (`handle_gc`).
        counts = _dry_run_counts(index, repo, drop_set=drop_set)
        split = index.count_gc_chunk_split(repo, list(drop_set))
        return GcCounts(
            snapshots=counts.snapshots,
            file_snapshots=counts.file_snapshots,
            edges=counts.edges,
            chunks=split.dropped,
        )

    # Real run: accumulate the *actual* deletions (commit drops + the
    # global orphan prune in `cleanup`), so the chunk figure is the true
    # reclamation — including orphans left by a forgotten repo.
    with index.session() as session:
        session.sweep()
        total = GcCounts()
        for sha in drop_set:
            total = total + session.drop_snapshot(at=repo.at(sha))
        total = total + session.cleanup(repo)
        if compact:
            session.compact()
    return total


def _run_orphans_only(
    index: Index, repo: Repo, *, dry_run: bool, compact: bool = False
) -> GcCounts:
    if dry_run:
        # ORPHANS dry-run is conservative: report zero rather than
        # adding a read-only counterpart to sweep_orphan_commits.
        return GcCounts()
    with index.session() as session:
        session.sweep()
        total = session.cleanup(repo)
        if compact:
            session.compact()
    return total


def _resolve_keep_set(
    index: Index,
    repo: Repo,
    *,
    mode: GcMode,
    refs: list[str],
) -> set[str]:
    """Return SHAs that must be preserved for this mode."""
    repo_path = repo.repo_path
    head = head_sha(repo_path)
    if mode is GcMode.HEAD_ONLY:
        return {head}
    if mode is GcMode.KEEP:
        return {head, *resolve_refs_to_shas(repo_path, refs)}
    if mode is GcMode.WATCHED_ONLY:
        return {head, *_watched_shas(index, repo)}
    # WATCHED (default): all local refs plus the resolved watch set.
    return {
        head,
        *local_ref_shas(repo_path, _KEPT_REF_PREFIXES),
        *_watched_shas(index, repo),
    }


def _watched_shas(index: Index, repo: Repo) -> set[str]:
    """Resolve the repo's watched refs to SHAs, skipping unresolvable ones."""
    out: set[str] = set()
    for ref in index.watched_refs(repo):
        try:
            out.add(
                resolve_ref(repo.repo_path, ref, max_untracked_files=config.max_untracked_files)
            )
        except RbtrError:
            # A watched ref that no longer resolves (deleted branch)
            # keeps nothing; its old commit becomes collectable.
            continue
    return out


def _resolve_drop_set(index: Index, repo: Repo, *, keep_set: set[str]) -> set[str]:
    """Return indexed SHAs to drop: everything outside *keep_set*."""
    indexed = {s.snapshot_sha for s in index.indexed_snapshots(repo)}
    # Protect the current worktree tree SHA from GC.  Tree SHAs
    # are never reachable from a git ref (they're tree objects,
    # not commits) so indexed - keep_set would include them.
    # The *current* tree SHA should be preserved; stale ones are
    # dropped by the server's post-build cleanup.
    current_wt = worktree_tree_sha(repo.repo_path, max_untracked_files=config.max_untracked_files)
    if current_wt is not None:
        indexed.discard(current_wt)
    return indexed - keep_set


def _dry_run_counts(index: Index, repo: Repo, *, drop_set: set[str]) -> GcCounts:
    """Compute commit / snapshot / edge counts without writing.

    The chunk figures are computed separately by `run_gc` via
    `count_gc_chunk_split` (a read-only reference-graph query that
    needs no drop simulation), so this returns `chunks=0` and the
    caller overrides it.
    """
    total = GcCounts(snapshots=len(drop_set))
    file_snapshots = 0
    edges = 0
    for sha in drop_set:
        file_snapshots += index.count_file_snapshots(at=repo.at(sha))
        edges += index.count_edges(at=repo.at(sha))
    return GcCounts(
        snapshots=total.snapshots,
        file_snapshots=file_snapshots,
        edges=edges,
        chunks=0,
    )
