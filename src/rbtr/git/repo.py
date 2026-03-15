"""Repository lifecycle — open, status, branches, remotes, fetch."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pygit2

from rbtr.exceptions import RbtrError
from rbtr.models import BranchSummary

log = logging.getLogger(__name__)


def find_git_root(start: str = ".") -> str | None:
    """Return the worktree root, or `None` if not in a repo."""
    git_dir = pygit2.discover_repository(start)
    if git_dir is None:
        return None
    return str(Path(git_dir).resolve().parent)


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


def default_branch(repo: pygit2.Repository) -> str:
    """Return the name of the repository's default branch.

    Tries `refs/remotes/origin/HEAD` first (set by `git clone`),
    then falls back to `main` or `master` if either exists locally.
    Returns `"main"` as a last resort.
    """
    # Try the symbolic ref that git clone sets.
    try:
        ref = repo.references.get("refs/remotes/origin/HEAD")
        if ref is not None:
            target = ref.resolve().shorthand
            # target looks like "origin/main" — strip the remote prefix.
            return target.split("/", 1)[-1]
    except (pygit2.GitError, AttributeError):
        pass

    # Fall back to well-known names.
    for name in ("main", "master"):
        if name in repo.branches.local:
            return name

    return "main"


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
