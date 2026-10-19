"""Editing what the index tracks: watch sets and registered repos.

`unwatch_refs` drops refs a caller names, `remove_stale_refs` drops
the ones git can no longer resolve, and `forget_stale_repos` drops
repos whose checkout is gone.  Each writes metadata only; see
ARCHITECTURE, "Watch-set lifecycle", for how they relate to
reclamation and why a cross-repo pass writes one session per repo.
"""

from __future__ import annotations

from rbtr.domain.models import RefsByRepo, Repo
from rbtr.errors import RbtrError
from rbtr.git import HEAD_REF, normalise_repo_path, resolve_refs_to_shas
from rbtr.index.index import Index


def unwatch_refs(index: Index, *, repo_path: str, refs: list[str], dry_run: bool) -> RefsByRepo:
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
    repo = index.find_repo(repo_path)
    if repo is None:
        return {}
    if not dry_run:
        with index.session() as session:
            session.remove_watched_refs(repo, refs)
    return {repo_path: refs}


def remove_stale_refs(index: Index, repos: list[Repo], *, dry_run: bool) -> RefsByRepo:
    """Stop watching the refs git can no longer resolve in *repos*, keyed by repo path.

    `HEAD` resolves for as long as the repo does and is never removed,
    and a repo whose path has vanished is left entirely alone: git
    cannot answer for it, and every ref it watches would look stale.
    Under *dry_run* nothing is written.
    """
    removed: RefsByRepo = {}
    for repo in repos:
        if not _resolves(repo):
            continue
        stale = _unresolvable(index, repo)
        if not stale:
            continue
        removed[repo.repo_path] = stale
        if not dry_run:
            with index.session() as session:
                session.remove_watched_refs(repo, stale)
    return removed


def forget_stale_repos(index: Index, *, dry_run: bool) -> list[Repo]:
    """Forget every registered repo whose path no longer resolves.

    Found by enumeration, since a removed checkout's path cannot be
    normalised and so cannot be named by a caller.  Forgetting is
    metadata-only: the chunks it orphans are reclaimed by the next GC.
    Under *dry_run* nothing is written.
    """
    gone = [repo for repo in index.repos() if not _resolves(repo)]
    if not dry_run:
        for repo in gone:
            with index.session() as session:
                session.forget_repo(repo)
    return gone


def _resolves(repo: Repo) -> bool:
    """Whether the repo is still where the index says it is."""
    try:
        normalise_repo_path(repo.repo_path)
    except RbtrError:
        return False
    return True


def _unresolvable(index: Index, repo: Repo) -> list[str]:
    """The repo's watched refs git can no longer resolve, `HEAD` aside.

    A full SHA is asked of git too, so one whose commit git has lost is
    stale.
    """
    out: list[str] = []
    for ref in index.watched_refs(repo):
        if ref == HEAD_REF:
            continue
        try:
            resolve_refs_to_shas(repo.repo_path, [ref])
        except RbtrError:
            out.append(ref)
    return out
