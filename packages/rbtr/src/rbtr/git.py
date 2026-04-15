"""Git operations for the code index.

All repository interaction goes through pygit2's object store —
no working-tree checkout is needed. This module provides the
minimal surface the index requires: opening a repo, resolving
refs, walking trees, listing files, and detecting changes.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pathspec
import pygit2

from rbtr.errors import RbtrError

if TYPE_CHECKING:
    from collections.abc import Iterator


# ── Data types ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class FileEntry:
    """A file in the git tree with its decoded content.

    Produced by `list_files` for every indexable file. The
    `blob_sha` uniquely identifies the content and is used as
    the cache key in the DuckDB index — if a file's blob SHA
    hasn't changed, its chunks don't need re-parsing.
    """

    path: str
    blob_sha: str
    content: bytes


# ── Repo open ────────────────────────────────────────────────────────


def open_repo(path: str = ".") -> pygit2.Repository:
    """Open the git repository containing *path*.

    Walks upward from *path* to find the `.git` directory.
    Raises `RbtrError` if *path* is not inside a git repository.
    """
    discovered = pygit2.discover_repository(path)
    if discovered is None:
        msg = f"Not a git repository: {path}"
        raise RbtrError(msg)
    return pygit2.Repository(discovered)


# ── Path filtering ───────────────────────────────────────────────────


def _matches_globs(path: str, patterns: list[str]) -> bool:
    """Return whether *path* matches any glob in *patterns*.

    A literal pattern (no wildcards) also matches child paths —
    e.g. pattern `".rbtr/"` matches `".rbtr/index/data.db"`.
    Trailing slashes on directory patterns are handled correctly.
    """
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        prefix = pat.rstrip("/")
        if not any(c in pat for c in "*?[") and (path == prefix or path.startswith(prefix + "/")):
            return True
    return False


def is_binary(data: bytes, sample_size: int = 8192) -> bool:
    """Heuristic binary detection.

    Returns `True` if a null byte appears in the first
    *sample_size* bytes of *data*.
    """
    return b"\x00" in data[:sample_size]


# ── Ref resolution ───────────────────────────────────────────────────


def resolve_commit(repo: pygit2.Repository, ref: str) -> pygit2.Commit:
    """Resolve *ref* to a `pygit2.Commit`.

    Accepts a SHA, branch name, or tag. Tries the bare ref first,
    then `origin/<ref>` for remote-tracking branches (common when
    the branch only exists on the remote).

    Raises `KeyError` if neither resolves.
    """
    for candidate in (ref, f"origin/{ref}"):
        try:
            obj = repo.revparse_single(candidate)
            return obj.peel(pygit2.Commit)
        except (KeyError, ValueError):
            continue
    msg = f"Cannot resolve ref '{ref}' (tried local and origin remote)"
    raise KeyError(msg)


# ── Tree walking ─────────────────────────────────────────────────────


def walk_tree(
    repo: pygit2.Repository,
    tree: pygit2.Tree,
    prefix: str,
) -> Iterator[tuple[str, pygit2.Blob]]:
    """Recursively walk *tree*, yielding `(path, blob)` pairs.

    Descends into subtrees. Skips non-blob entries (submodules,
    symlinks). Paths are built by joining *prefix* with each
    entry name.
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


def list_files(
    repo: pygit2.Repository,
    ref: str,
    *,
    max_file_size: int,
    ignore: pathspec.PathSpec | None = None,
) -> Iterator[FileEntry]:
    """Yield every indexable file in the tree at *ref*.

    Applies layered filtering:

    1. `.gitignore` rules via `repo.path_is_ignored`.
    2. `.rbtrignore` rules via *ignore* (when provided).
    3. Files larger than *max_file_size* are skipped.
    4. Binary files (null-byte heuristic) are skipped.
    """
    commit = resolve_commit(repo, ref)
    tree = commit.tree

    for entry in walk_tree(repo, tree, ""):
        path, blob = entry
        if repo.path_is_ignored(path):
            continue
        if ignore is not None and ignore.match_file(path):
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


def read_blob(
    repo: pygit2.Repository,
    ref: str,
    path: str,
) -> pygit2.Blob | None:
    """Read a single blob from the object store.

    Returns the `pygit2.Blob` for *path* at *ref*, or `None` if
    the file doesn't exist at that ref. Uses tree-path lookup
    for O(log n) access.
    """
    try:
        commit = resolve_commit(repo, ref)
    except KeyError:
        return None
    try:
        entry = commit.tree[path]
    except KeyError:
        return None
    obj = repo.get(entry.id)
    if isinstance(obj, pygit2.Blob):
        return obj
    return None


# ── Changed files ────────────────────────────────────────────────────


def changed_files(
    repo: pygit2.Repository,
    base_ref: str,
    head_ref: str,
) -> set[str]:
    """Return file paths that differ between *base_ref* and *head_ref*.

    Includes added, modified, and deleted files. Used by the index
    to determine which files need re-parsing during incremental
    updates.
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
