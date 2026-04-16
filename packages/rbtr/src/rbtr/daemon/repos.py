"""Repo manager — resolves repo paths to store repo_ids.

Wraps `IndexStore.register_repo()` with caching so repeated
requests for the same repo don't hit the database.
"""

from __future__ import annotations

from rbtr.index.store import IndexStore


class RepoManager:
    """Maps repo paths to integer repo_ids.

    Registration is implicit — the first access for a repo
    path creates the entry in the `repos` table.
    """

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        self._cache: dict[str, int] = {}

    @property
    def store(self) -> IndexStore:
        """The underlying IndexStore."""
        return self._store

    def resolve(self, repo: str) -> int:
        """Return the repo_id for *repo*, registering if needed."""
        cached = self._cache.get(repo)
        if cached is not None:
            return cached
        repo_id = self._store.register_repo(repo)
        self._cache[repo] = repo_id
        return repo_id
