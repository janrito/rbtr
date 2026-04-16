"""Ref watcher — polls git repos for HEAD changes.

Pure polling logic with no threads or daemon dependencies.
The daemon server runs `poll()` periodically and acts on
the returned changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pygit2

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefChange:
    """A detected HEAD change in a watched repo."""

    repo_path: str
    old_ref: str
    new_ref: str


def _read_head(repo_path: str) -> str | None:
    """Read the current HEAD SHA for a repo, or None on error."""
    try:
        repo = pygit2.Repository(repo_path)
        if repo.head_is_unborn:
            return None
        return str(repo.head.target)
    except Exception:
        log.warning("Failed to read HEAD for %s", repo_path, exc_info=True)
        return None


class RefWatcher:
    """Watches registered repos for HEAD changes.

    Call `register()` to add repos, then `poll()` periodically.
    `poll()` returns a list of repos whose HEAD changed since
    the last poll.
    """

    def __init__(self) -> None:
        self._refs: dict[str, str | None] = {}

    def register(self, repo_path: str) -> None:
        """Start watching a repo. Records current HEAD."""
        self._refs[repo_path] = _read_head(repo_path)

    def unregister(self, repo_path: str) -> None:
        """Stop watching a repo."""
        self._refs.pop(repo_path, None)

    def repos(self) -> list[str]:
        """Return all watched repo paths."""
        return list(self._refs)

    def poll(self) -> list[RefChange]:
        """Check all watched repos for HEAD changes.

        Returns a list of changes since the last poll.
        Updates stored refs for changed repos.
        """
        changes: list[RefChange] = []
        for repo_path, last_ref in self._refs.items():
            current = _read_head(repo_path)
            if current is not None and current != last_ref:
                if last_ref is not None:
                    changes.append(
                        RefChange(
                            repo_path=repo_path,
                            old_ref=last_ref,
                            new_ref=current,
                        )
                    )
                self._refs[repo_path] = current
        return changes
