"""Local git repository operations for rbtr."""

import re
from datetime import UTC, datetime

import pygit2

from rbtr.exceptions import RbtrError
from rbtr.models import BranchSummary


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


def parse_github_remote(repo: pygit2.Repository) -> tuple[str, str]:
    """Extract (owner, repo_name) from the GitHub remote URL.

    Tries 'origin' first, then falls back to the first remote that looks like GitHub.
    """
    remotes: list[pygit2.Remote] = list(repo.remotes)

    # Try origin first
    for remote in remotes:
        if remote.name == "origin" and remote.url is not None:
            result = _parse_github_url(remote.url)
            if result is not None:
                return result

    # Fall back to any GitHub remote
    for remote in remotes:
        if remote.url is not None:
            result = _parse_github_url(remote.url)
            if result is not None:
                return result

    raise RbtrError("No GitHub remote found. rbtr requires a repository with a GitHub remote.")


_GITHUB_PATTERNS = [
    # SSH: git@github.com:owner/repo.git
    re.compile(r"git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"),
    # HTTPS: https://github.com/owner/repo.git
    re.compile(r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"),
]


def _parse_github_url(url: str) -> tuple[str, str] | None:
    for pattern in _GITHUB_PATTERNS:
        m = pattern.match(url)
        if m:
            return m.group("owner"), m.group("repo")
    return None


def default_branch(repo: pygit2.Repository) -> str:
    """Return the name of the repository's default branch.

    Tries ``refs/remotes/origin/HEAD`` first (set by ``git clone``),
    then falls back to ``main`` or ``master`` if either exists locally.
    Returns ``"main"`` as a last resort.
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
