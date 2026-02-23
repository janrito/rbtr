"""Object-store reading — resolve refs, walk trees, diffs, logs.

All reads go through the git object store — no working-tree checkout
is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pygit2
from pygit2.enums import SortMode

from rbtr.config import config
from rbtr.git.filters import is_binary, is_path_ignored

if TYPE_CHECKING:
    from collections.abc import Iterator


# ── Data types ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class FileEntry:
    """A file in the git tree with its content."""

    path: str
    blob_sha: str
    content: bytes


@dataclass(frozen=True)
class DiffStats:
    """Raw numeric statistics for a diff."""

    files_changed: int
    insertions: int
    deletions: int


@dataclass(frozen=True)
class DiffResult:
    """Unified diff output between two commits."""

    stats: DiffStats
    """Numeric change summary."""

    patch_lines: list[str]
    """Full patch text split into lines."""


@dataclass(frozen=True)
class LogEntry:
    """One commit in a log range."""

    sha: str
    """Full hex SHA of the commit."""

    author: str
    message: str


# ── Ref resolution ───────────────────────────────────────────────────


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


# ── Tree walking ─────────────────────────────────────────────────────


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


def read_blob(
    repo: pygit2.Repository,
    ref: str,
    path: str,
) -> pygit2.Blob | None:
    """Return the blob for *path* at *ref*, or ``None`` if not found.

    Uses tree path lookup for O(log n) access instead of walking
    the entire tree.
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


# ── Diffs ────────────────────────────────────────────────────────────


def diff_refs(
    repo: pygit2.Repository,
    base_ref: str,
    head_ref: str,
    *,
    path: str = "",
) -> DiffResult:
    """Compute a diff between two refs.

    When *path* is non-empty, restricts the diff to that file.

    Raises:
        KeyError: If either ref cannot be resolved.
    """
    base_commit = resolve_commit(repo, base_ref)
    head_commit = resolve_commit(repo, head_ref)
    d = repo.diff(base_commit, head_commit)
    return _build_diff_result(d, path)


def diff_single(
    repo: pygit2.Repository,
    ref: str,
    *,
    path: str = "",
) -> DiffResult:
    """Diff a single commit against its parent.

    Raises:
        KeyError: If *ref* cannot be resolved.
        ValueError: If the commit has no parent (initial commit)
            or the parent object is missing.
    """
    commit = resolve_commit(repo, ref)
    if not commit.parent_ids:
        msg = f"Commit {ref} has no parent (initial commit)."
        raise ValueError(msg)
    parent_obj = repo.get(commit.parent_ids[0])
    if parent_obj is None:
        msg = f"Parent of {ref} not found."
        raise ValueError(msg)
    parent = parent_obj.peel(pygit2.Commit)
    d = repo.diff(parent, commit)
    return _build_diff_result(d, path)


def _build_diff_result(d: pygit2.Diff, path: str) -> DiffResult:
    """Build a ``DiffResult`` from a pygit2 Diff.

    When *path* is given but has no changes, returns a ``DiffResult``
    with an empty ``patch_lines`` list and zeroed stats.
    """
    if path:
        patches = [
            p
            for p in d
            if p is not None and (p.delta.old_file.path == path or p.delta.new_file.path == path)
        ]
        if not patches:
            return DiffResult(
                stats=DiffStats(files_changed=0, insertions=0, deletions=0),
                patch_lines=[],
            )
        patch_text = "\n".join(p.text for p in patches if p.text)
        n_files = len(patches)
        insertions = sum(p.line_stats[1] for p in patches)
        deletions = sum(p.line_stats[2] for p in patches)
    else:
        raw = d.stats
        n_files = raw.files_changed
        insertions = raw.insertions
        deletions = raw.deletions
        patch_text = d.patch or "(empty diff)"

    return DiffResult(
        stats=DiffStats(files_changed=n_files, insertions=insertions, deletions=deletions),
        patch_lines=patch_text.splitlines(),
    )


# ── Changed files ────────────────────────────────────────────────────


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


# ── Commit log ───────────────────────────────────────────────────────


def commit_log_between(
    repo: pygit2.Repository,
    base_ref: str,
    head_ref: str,
) -> list[LogEntry]:
    """Return commits on *head_ref* not reachable from *base_ref*.

    Commits are in reverse chronological order.  Returns an empty
    list when the branches are identical.

    Raises:
        KeyError: If either ref cannot be resolved.
    """
    head_commit = resolve_commit(repo, head_ref)
    base_commit = resolve_commit(repo, base_ref)

    merge_base = repo.merge_base(head_commit.id, base_commit.id)
    stop_at = merge_base if merge_base else None

    entries: list[LogEntry] = []
    for commit in repo.walk(head_commit.id, SortMode.TOPOLOGICAL):
        if stop_at and commit.id == stop_at:
            break
        entries.append(
            LogEntry(
                sha=str(commit.id),
                author=commit.author.name,
                message=commit.message.strip().split("\n", 1)[0],
            )
        )
    return entries
