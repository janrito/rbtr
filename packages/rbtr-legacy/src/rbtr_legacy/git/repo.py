"""Repository lifecycle — open, status, branches, remotes, fetch."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pygit2

from rbtr_legacy.exceptions import RbtrError
from rbtr_legacy.models import BranchSummary

log = logging.getLogger(__name__)


def open_repo() -> pygit2.Repository:
    """Open the git repository containing the current directory."""
    path = pygit2.discover_repository(".")
    if path is None:
        raise RbtrError("rbtr must be run from inside a git repository.")
    return pygit2.Repository(path)


def require_clean(repo: pygit2.Repository) -> None:
    """Ensure the working tree has no uncommitted changes (staged or unstaged).

    Untracked files are allowed.
    """
    status = repo.status()
    dirty = {
        path
        for path, flags in status.items()
        if flags
        not in (
            pygit2.GIT_STATUS_IGNORED,
            pygit2.GIT_STATUS_WT_NEW,
        )
    }
    if dirty:
        raise RbtrError(
            "Working tree has uncommitted changes. Please commit or stash before running rbtr."
        )


def fetch_pr_refs(
    repo: pygit2.Repository,
    pr_number: int,
    base_branch: str,
) -> None:
    """Fetch a PR's head ref and its base branch from origin.

    Fetches both in a single call:

    - `refs/pull/<number>/head` — the PR head commit (works for
      forks and same-repo PRs).
    - `refs/heads/<base_branch>` → `refs/remotes/origin/<base_branch>`
      — keeps the remote-tracking ref for the base branch current so
      that diffs, commit logs, and changed-file lists reflect the
      real PR scope.

    Non-destructive: local branches and the working tree are never
    modified.

    Silently succeeds if the fetch fails (e.g. no network, auth
    error) — callers handle missing refs downstream.
    """
    try:
        remote = repo.remotes["origin"]
    except KeyError:
        return
    url = remote.url
    if url is None:
        return
    pr_ref = f"refs/pull/{pr_number}/head"
    refspecs = [
        f"+{pr_ref}:{pr_ref}",
        f"+refs/heads/{base_branch}:refs/remotes/origin/{base_branch}",
    ]
    callbacks = _make_fetch_callbacks(url)
    try:
        remote.fetch(refspecs, callbacks=callbacks)
    except pygit2.GitError as exc:
        log.debug("fetch_pr_refs(#%d, %s) failed: %s", pr_number, base_branch, exc)


def _make_fetch_callbacks(url: str) -> pygit2.RemoteCallbacks:
    """Build `RemoteCallbacks` with credentials suitable for *url*.

    SSH URLs (`git@…` or `ssh://…`) → `KeypairFromAgent`
    (delegates to the user's SSH agent).  Everything else uses
    pygit2's default credential discovery.
    """
    if url.startswith("git@") or url.startswith("ssh://"):
        return pygit2.RemoteCallbacks(
            credentials=pygit2.KeypairFromAgent("git"),
        )
    return pygit2.RemoteCallbacks()


def list_local_branches(repo: pygit2.Repository) -> list[BranchSummary]:
    """List local branches sorted by most recently committed first.

    Excludes HEAD and the current branch.
    """
    current = None
    if not repo.head_is_unborn:
        current = repo.head.shorthand

    results: list[BranchSummary] = []
    for name in repo.branches.local:
        if name == current:
            continue
        branch = repo.branches.local[name]
        commit = branch.peel(pygit2.Commit)
        results.append(
            BranchSummary(
                name=name,
                last_commit_sha=str(commit.id),
                last_commit_message=commit.message.split("\n", 1)[0],
                updated_at=datetime.fromtimestamp(commit.commit_time, tz=UTC),
            )
        )

    results.sort(key=lambda b: b.updated_at, reverse=True)
    return results
