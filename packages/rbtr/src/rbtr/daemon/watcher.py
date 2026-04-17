"""Ref watcher — polls git repos for HEAD changes.

Pure polling logic with no threads or daemon dependencies.
The daemon server runs `poll()` periodically and acts on
the returned changes.

The set of watched repos is derived from the index store's
`repos` table on startup — every repo that has ever been
indexed is watched. New repos join the watcher when their
first `build_index` request is handled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rbtr.git import read_head

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefChange:
    """A detected HEAD change in a watched repo."""

    repo_path: str
    old_ref: str
    new_ref: str


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
        self._refs[repo_path] = read_head(repo_path)

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
            current = read_head(repo_path)
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
