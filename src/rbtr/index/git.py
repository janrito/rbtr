"""Read file contents at a commit via pygit2.

All reads go through the git object store — no working-tree checkout
is needed.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pygit2

from rbtr.config import config

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class FileEntry:
    """A file in the git tree with its content."""

    path: str
    blob_sha: str
    content: bytes


def resolve_commit(repo: pygit2.Repository, ref: str) -> pygit2.Commit:
    """Resolve a ref (SHA, branch name, tag) to a Commit object.

    Tries the bare ref first, then ``origin/<ref>`` for remote
    branches (common when reviewing PRs whose head branch only
    exists on the remote).
    """
    for candidate in (ref, f"origin/{ref}"):
        try:
            obj = repo.revparse_single(candidate)
            return obj.peel(pygit2.Commit)
        except (KeyError, ValueError):
            continue
    msg = f"Cannot resolve ref '{ref}' (tried local and origin remote)"
    raise KeyError(msg)


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


def list_files(
    repo: pygit2.Repository,
    ref: str,
    *,
    max_file_size: int | None = None,
) -> Iterator[FileEntry]:
    """Yield every indexable file in the tree at *ref*.

    Filtering is layered:

    1. ``config.index.include`` globs force-include paths even if
       they match ``.gitignore`` (e.g. a vendored file you want indexed).
    2. The repo's ``.gitignore`` (checked via ``repo.path_is_ignored``).
    3. ``config.index.extend_exclude`` globs from user config.

    Binary files and files larger than *max_file_size* are also skipped.
    Defaults come from ``config.index``.
    """
    if max_file_size is None:
        max_file_size = config.index.max_file_size
    commit = resolve_commit(repo, ref)
    tree = commit.tree

    for entry in walk_tree(repo, tree, ""):
        path, blob = entry
        if is_path_ignored(path, repo):
            continue
        if blob.size > max_file_size:
            continue
        data = blob.data
        if is_binary(data):
            continue
        yield FileEntry(
            path=path,
            blob_sha=str(blob.id),
            content=data,
        )


def walk_tree(
    repo: pygit2.Repository,
    tree: pygit2.Tree,
    prefix: str,
) -> Iterator[tuple[str, pygit2.Blob]]:
    """Recursively walk a git tree, yielding ``(path, blob)`` pairs.

    Uses ``entry.type_str`` to dispatch without fetching the object
    first — avoids an ``isinstance`` check on every entry.
    """
    for entry in tree:
        path = f"{prefix}{entry.name}" if not prefix else f"{prefix}/{entry.name}"
        match entry.type_str:
            case "tree":
                obj = repo.get(entry.id)
                if isinstance(obj, pygit2.Tree):
                    yield from walk_tree(repo, obj, path)
            case "blob":
                obj = repo.get(entry.id)
                if isinstance(obj, pygit2.Blob):
                    yield path, obj


def changed_files(
    repo: pygit2.Repository,
    base_ref: str,
    head_ref: str,
) -> set[str]:
    """Return the set of file paths that differ between two refs.

    Includes added, modified, and deleted files.
    """
    base_commit = resolve_commit(repo, base_ref)
    head_commit = resolve_commit(repo, head_ref)
    diff = repo.diff(base_commit, head_commit)

    paths: set[str] = set()
    for patch in diff:
        if patch is None:
            continue
        delta = patch.delta
        if delta.old_file.path:
            paths.add(delta.old_file.path)
        if delta.new_file.path:
            paths.add(delta.new_file.path)
    return paths
