"""Garbage collection for the code index.

One entry point, ``run_gc``, executes every supported mode
(``GcMode``) over a single repo.  The caller supplies a git
``pygit2.Repository`` for ref resolution and an ``IndexStore`` for
the actual writes.  Both daemon handlers and the inline CLI
fallback call this function; the schema of its inputs is the
single source of truth for what ``rbtr gc`` can do.

Modes
-----

HEAD_ONLY
    Keep the repo's current HEAD; drop every other indexed commit.
KEEP_REFS
    Keep HEAD plus every local branch / tag / note.  Only refs
    actually present in ``indexed_commits`` are kept; we never
    trigger new indexing from a GC call.
KEEP
    Keep the union of HEAD and the caller-supplied refs; drop rest.
DROP
    Drop only the caller-supplied refs; keep the rest.
ORPHANS
    Do not drop any indexed commits.  Only sweep residue from
    crashed builds (`sweep_orphan_commits`, which also removes
    orphaned chunks).
"""

from __future__ import annotations

import logging

import pygit2

from rbtr.daemon.messages import GcMode
from rbtr.errors import RbtrError
from rbtr.git import resolve_commit
from rbtr.index.store import GcCounts, IndexStore

log = logging.getLogger(__name__)

# Reference namespaces that contribute to ``KEEP_REFS``.
_KEPT_REF_PREFIXES: tuple[str, ...] = (
    "refs/heads/",
    "refs/tags/",
    "refs/notes/",
)


def run_gc(
    store: IndexStore,
    repo: pygit2.Repository,
    repo_id: int,
    *,
    mode: GcMode,
    refs: list[str],
    dry_run: bool,
) -> GcCounts:
    """Execute a garbage-collection operation.

    Returns accumulated ``GcCounts`` across every deletion performed.
    When *dry_run* is true, computes the counts without writing to
    the database.
    """
    if mode is GcMode.ORPHANS:
        return _run_orphans_only(store, repo_id, dry_run=dry_run)

    keep_set = _resolve_keep_set(repo, mode=mode, refs=refs)
    drop_set = _resolve_drop_set(
        store,
        repo=repo,
        repo_id=repo_id,
        mode=mode,
        refs=refs,
        keep_set=keep_set,
    )

    if dry_run:
        return _dry_run_counts(store, repo_id, drop_set=drop_set)

    total = GcCounts()
    for sha in drop_set:
        total = total + store.drop_commit(repo_id, sha)
    # Residue sweep in case the index had leftover rows pre-existing
    # from a crashed build.
    total = total + store.sweep_orphan_commits(repo_id)
    return total


def _run_orphans_only(store: IndexStore, repo_id: int, *, dry_run: bool) -> GcCounts:
    if dry_run:
        # ORPHANS dry-run is conservative: report zero rather than
        # adding a read-only counterpart to sweep_orphan_commits.
        return GcCounts()
    return store.sweep_orphan_commits(repo_id)


def _resolve_keep_set(
    repo: pygit2.Repository,
    *,
    mode: GcMode,
    refs: list[str],
) -> set[str]:
    """Return SHAs that must be preserved for this mode."""
    head_sha = _head_sha(repo)
    if mode is GcMode.HEAD_ONLY:
        return {head_sha}
    if mode is GcMode.KEEP_REFS:
        return {head_sha, *_local_ref_shas(repo)}
    if mode is GcMode.KEEP:
        return {head_sha, *_resolve_refs(repo, refs)}
    # DROP has no keep set — every indexed commit is eligible unless
    # explicitly listed in ``refs``.
    return set()


def _resolve_drop_set(
    store: IndexStore,
    *,
    repo: pygit2.Repository,
    repo_id: int,
    mode: GcMode,
    refs: list[str],
    keep_set: set[str],
) -> set[str]:
    """Return SHAs to drop given *mode* and the resolved *keep_set*."""
    indexed = {sha for sha, _at in store.list_indexed_commits(repo_id)}
    if mode is GcMode.DROP:
        return _resolve_refs(repo, refs) & indexed
    return indexed - keep_set


def _dry_run_counts(store: IndexStore, repo_id: int, *, drop_set: set[str]) -> GcCounts:
    """Compute counts without writing.

    Accurate for commits / snapshots / edges.  The ``chunks`` field
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


def _head_sha(repo: pygit2.Repository) -> str:
    if repo.head_is_unborn:
        msg = "Cannot GC a repo with no commits (HEAD is unborn)"
        raise RbtrError(msg)
    return str(repo.head.target)


def _local_ref_shas(repo: pygit2.Repository) -> set[str]:
    out: set[str] = set()
    for name in repo.references:
        if not name.startswith(_KEPT_REF_PREFIXES):
            continue
        try:
            out.add(str(repo.references[name].peel(pygit2.Commit).id))
        except Exception:
            # Malformed refs are ignored; GC should still complete.
            log.debug("Skipping malformed ref %s", name, exc_info=True)
            continue
    return out


def _resolve_refs(repo: pygit2.Repository, refs: list[str]) -> set[str]:
    out: set[str] = set()
    for ref in refs:
        try:
            out.add(str(resolve_commit(repo, ref).id))
        except KeyError as exc:
            msg = f"Cannot resolve ref {ref!r}: {exc}"
            raise RbtrError(msg) from exc
    return out
