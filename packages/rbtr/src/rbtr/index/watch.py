"""Editing what the index tracks: watch sets and registered repos.

Three operations, kept apart because they answer different questions.
`unwatch_refs` drops refs a caller names. `remove_stale_refs` asks a
live repo which of its watched refs git can still resolve.
`forget_stale_repos` asks whether a registered repo is there at all.
Each is the one place its answer is decided; a caller wanting two runs
both.

All three are bookkeeping about what the index *tracks*.  Reclaiming
the storage those entries used is `rbtr.index.gc`'s job, which is why
forgetting a repo here does not sweep its chunks.

A pass covering several repos writes one session per repo rather than
one spanning all of them: a repo is the consistency boundary, and the
work is idempotent, so a pass interrupted part-way is finished by
running it again.
"""

from __future__ import annotations

from rbtr.domain.models import RefsByRepo, Repo, Scope
from rbtr.errors import RbtrError
from rbtr.git import HEAD_REF, normalise_repo_path, resolve_ref
from rbtr.index.store import IndexStore


def unwatch_refs(
    store: IndexStore, *, repo_path: str, refs: list[str], dry_run: bool
) -> RefsByRepo:
    """Stop watching the named refs in one repo.

    `HEAD` is refused **before any delete**, so a call naming it
    alongside others changes nothing.  Refs that were not watched are
    reported as removed all the same: the caller asked for them to be
    gone, and they are.  A repo that was never indexed watches nothing,
    so nothing is removed.  Under *dry_run* nothing is written.
    """
    if HEAD_REF in refs:
        msg = "HEAD cannot be removed from the watch set"
        raise RbtrError(msg)
    repo_id = store.get_repo_id(repo_path)
    if repo_id is None:
        return {}
    if not dry_run:
        with store.session() as session:
            session.remove_watched_refs(repo_id, refs)
    return {repo_path: refs}


def remove_stale_refs(
    store: IndexStore, *, repo_path: str, scope: Scope, dry_run: bool
) -> RefsByRepo:
    """Stop watching refs git can no longer resolve, keyed by repo path.

    Covers the repo at *repo_path*, or every registered repo under
    `Scope.ALL`.  `HEAD` resolves for as long as the repo does and is
    never removed, and a repo whose path has vanished is left entirely
    alone: git cannot answer for it, and every ref it watches would
    look stale.  Under *dry_run* nothing is written.

    Raises `RbtrError` if *repo_path* names a repo that was never
    indexed.
    """
    scoped_id = store.resolve_repo(repo_path) if scope is Scope.WORKSPACE else None
    removed: RefsByRepo = {}
    for repo in store.list_repos():
        if (scoped_id is not None and repo.repo_id != scoped_id) or not _resolves(repo):
            continue
        stale = _unresolvable(store, repo)
        if not stale:
            continue
        removed[repo.repo_path] = stale
        if not dry_run:
            with store.session() as session:
                session.remove_watched_refs(repo.repo_id, stale)
    return removed


def forget_stale_repos(store: IndexStore, *, dry_run: bool) -> list[Repo]:
    """Forget every registered repo whose path no longer resolves.

    Found by enumeration, since a removed checkout's path cannot be
    normalised and so cannot be named by a caller.  Forgetting is
    metadata-only: the chunks it orphans are reclaimed by the next GC.
    Under *dry_run* nothing is written.
    """
    gone = [repo for repo in store.list_repos() if not _resolves(repo)]
    if not dry_run:
        for repo in gone:
            with store.session() as session:
                session.forget_repo(repo.repo_id)
    return gone


def _resolves(repo: Repo) -> bool:
    """Whether the repo is still where the index says it is."""
    try:
        normalise_repo_path(repo.repo_path)
    except RbtrError:
        return False
    return True


def _unresolvable(store: IndexStore, repo: Repo) -> list[str]:
    """The repo's watched refs git can no longer resolve, `HEAD` aside."""
    out: list[str] = []
    for ref in store.list_watched_refs(repo.repo_id):
        if ref == HEAD_REF:
            continue
        try:
            resolve_ref(repo.repo_path, ref)
        except RbtrError:
            out.append(ref)
    return out
