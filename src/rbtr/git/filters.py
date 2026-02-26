"""Path filtering — gitignore, globs, binary detection."""

from __future__ import annotations

import fnmatch

import pygit2


def _matches_globs(path: str, patterns: list[str]) -> bool:
    """Check whether *path* matches any of the given globs.

    A literal pattern (no ``*``, ``?``, or ``[``) also matches
    any child path — e.g. pattern ``".rbtr/"`` matches
    ``".rbtr/index/data.db"``.  Trailing slashes on directory
    patterns are handled correctly.
    """
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        # Treat literal patterns as directory prefixes too.
        # Strip trailing slash so ".rbtr/" doesn't become ".rbtr//".
        prefix = pat.rstrip("/")
        if not any(c in pat for c in "*?[") and (path == prefix or path.startswith(prefix + "/")):
            return True
    return False


def is_path_ignored(
    path: str,
    repo: pygit2.Repository | None,
    *,
    include: list[str],
    exclude: list[str],
) -> bool:
    """Check whether *path* should be excluded from file tools.

    Applies a three-layer filter:

    1. *include* globs force-include (override gitignore and exclude).
    2. ``.gitignore`` via ``repo.path_is_ignored`` (when *repo* is
       available).
    3. *exclude* globs.
    """
    forced = bool(include) and _matches_globs(path, include)
    if forced:
        return False
    if repo is not None and repo.path_is_ignored(path):
        return True
    return _matches_globs(path, exclude)


def is_binary(data: bytes, sample_size: int = 8192) -> bool:
    """Heuristic: a file is binary if the first *sample_size* bytes
    contain a null byte.
    """
    return b"\x00" in data[:sample_size]
