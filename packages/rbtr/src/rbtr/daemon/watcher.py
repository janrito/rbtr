"""Ref watcher — detects repos whose HEAD is not indexed.

The watcher is stateless. On each poll it reads the current HEAD
of every registered repo and asks the index store whether that
exact commit has been marked as fully indexed. Every commit the
store doesn't know about is reported as stale; callers (the daemon
worker loop) decide what to do next.

No in-memory state means no startup seeding, no register/unregister
API, and no drift between the watcher and the index: the store's
``indexed_commits`` table is the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from rbtr.git import read_head
from rbtr.index.store import IndexStore


@dataclass(frozen=True)
class StaleHead:
    """A repo whose current HEAD is not recorded in ``indexed_commits``."""

    repo_path: str
    new_ref: str


def poll(store: IndexStore) -> list[StaleHead]:
    """Return every repo whose current HEAD has not been fully indexed.

    Iterates over registered repos (``store.list_repos``), reads each
    one's current HEAD with ``git.read_head``, and yields a ``StaleHead``
    for any commit that has no ``indexed_commits`` row. Repos whose
    HEAD cannot be read (unborn, missing path, permission error) are
    silently skipped.
    """
    out: list[StaleHead] = []
    for repo_id, path in store.list_repos():
        current = read_head(path)
        if current is None:
            continue
        if store.has_indexed(repo_id, current):
            continue
        out.append(StaleHead(repo_path=path, new_ref=current))
    return out
