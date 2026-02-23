"""Path filtering — gitignore, globs, binary detection."""

from __future__ import annotations

import fnmatch

import pygit2

from rbtr.config import config


def _matches_globs(path: str, patterns: list[str]) -> bool:
    """Check whether *path* matches any of the given globs.

    A literal pattern (no ``*``, ``?``, or ``[``) also matches
    any child path — e.g. pattern ``".rbtr/index"`` matches
    ``".rbtr/index/data.db"``.
    """
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        # Treat literal patterns as directory prefixes too.
        if not any(c in pat for c in "*?[") and path.startswith(pat + "/"):
            return True
    return False


def is_path_ignored(
    path: str,
    repo: pygit2.Repository | None = None,
) -> bool:
    """Check whether *path* should be excluded from file tools.

    Applies the same three-layer filter as the indexer:

    1. ``config.index.include`` force-includes (overrides gitignore
       and extend_exclude).
    2. ``.gitignore`` via ``repo.path_is_ignored`` (when *repo* is
       available).
    3. ``config.index.extend_exclude`` globs.
    """
    include = config.index.include
    extend = config.index.extend_exclude
    forced = bool(include) and _matches_globs(path, include)
    if forced:
        return False
    if repo is not None and repo.path_is_ignored(path):
        return True
    return _matches_globs(path, extend)


def is_binary(data: bytes, sample_size: int = 8192) -> bool:
    """Heuristic: a file is binary if the first *sample_size* bytes
    contain a null byte.
    """
    return b"\x00" in data[:sample_size]
