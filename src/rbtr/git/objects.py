"""Object-store reading — resolve refs, walk trees, diffs, logs.

All reads go through the git object store — no working-tree checkout
is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pygit2
from pygit2.enums import SortMode

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
    max_file_size: int,
    include: list[str],
    exclude: list[str],
) -> Iterator[FileEntry]:
    """Yield every indexable file in the tree at *ref*.

    Filtering is layered:

    1. *include* globs force-include paths even if they match
       ``.gitignore`` (e.g. a vendored file you want indexed).
    2. The repo's ``.gitignore`` (checked via ``repo.path_is_ignored``).
    3. *exclude* globs.

    Binary files and files larger than *max_file_size* are also skipped.
    """
    commit = resolve_commit(repo, ref)
    tree = commit.tree

    for entry in walk_tree(repo, tree, ""):
        path, blob = entry
        if is_path_ignored(path, repo, include=include, exclude=exclude):
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


# ── Anchor resolution ────────────────────────────────────────────────


def resolve_anchor(
    repo: pygit2.Repository,
    ref: str,
    path: str,
    anchor: str,
) -> int | str:
    """Find *anchor* in the file at *ref*/*path* and return the line number.

    Returns the 1-indexed line number of the **last line** of the
    match (suitable for GitHub's ``line`` parameter on single-line
    and multi-line comments).

    Returns an error string when the anchor cannot be resolved:
    file not found, binary file, zero matches, or multiple matches.
    """
    blob = read_blob(repo, ref, path)
    if blob is None:
        return f"File '{path}' not found at ref '{ref}'."
    if is_binary(blob.data):
        return f"File '{path}' is binary — cannot resolve anchor."
    text = blob.data.decode(errors="replace")
    if not anchor:
        return "Anchor text is empty."

    # Find all occurrences of the anchor substring.
    matches: list[int] = []
    start = 0
    while True:
        idx = text.find(anchor, start)
        if idx == -1:
            break
        # Line number of the last line of the match.
        line = text[:idx].count("\n") + anchor.count("\n") + 1
        matches.append(line)
        start = idx + 1

    if not matches:
        return (
            f"Anchor text not found in '{path}' at ref '{ref}'. "
            f"Make sure the text is an exact substring of the file content."
        )
    if len(matches) > 1:
        locations = ", ".join(f"line {m}" for m in matches)
        return (
            f"Anchor text matches {len(matches)} locations in '{path}': {locations}. "
            f"Use a longer or more unique snippet."
        )
    return matches[0]


# ── Line translation ─────────────────────────────────────────────────


def translate_line(
    repo: pygit2.Repository,
    path: str,
    old_ref: str,
    new_ref: str,
    old_line: int,
) -> int | None:
    """Translate *old_line* from *old_ref* to *new_ref* via diff hunks.

    Returns the corresponding line number in *new_ref*, or ``None``
    if the line was deleted.

    The algorithm walks hunks sequentially, accumulating an offset
    (insertions minus deletions).  For each hunk:

    - Lines **before** the hunk: apply the accumulated offset.
    - Lines **inside** the hunk: scan individual diff lines.
      Context lines map via ``new_lineno``; deleted lines
      return ``None``.
    - Lines **after all hunks**: apply the final offset.

    If *path* has no changes between the two refs, returns
    *old_line* unchanged (identity).  If *path* was deleted,
    returns ``None``.
    """
    old_commit = resolve_commit(repo, old_ref)
    new_commit = resolve_commit(repo, new_ref)
    d = repo.diff(old_commit, new_commit)

    # Find the patch for this file.
    patch: pygit2.Patch | None = None
    for p in d:
        if p is None:
            continue
        delta = p.delta
        if delta.old_file.path == path or delta.new_file.path == path:
            patch = p
            break

    if patch is None:
        # File unchanged between refs — identity mapping.
        return old_line

    # File was deleted entirely.
    if patch.delta.status == pygit2.GIT_DELTA_DELETED:
        return None

    offset = 0
    for hunk in patch.hunks:
        hunk_old_start = hunk.old_start
        hunk_old_end = hunk_old_start + hunk.old_lines - 1

        if old_line < hunk_old_start:
            # Before this hunk — apply accumulated offset.
            return old_line + offset

        if old_line <= hunk_old_end:
            # Inside this hunk — scan individual lines.
            for diff_line in hunk.lines:
                if diff_line.old_lineno == old_line:
                    if diff_line.new_lineno >= 0:
                        return diff_line.new_lineno
                    # Deleted line (old exists, new == -1).
                    return None
            # old_line is in the hunk range but not in any diff line
            # (shouldn't happen with well-formed diffs).
            return None  # pragma: no cover

        # Past this hunk — accumulate offset and continue.
        offset += hunk.new_lines - hunk.old_lines

    # After all hunks — apply final offset.
    return old_line + offset


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


# ── Diff line ranges (for review comment validation) ─────────────────

type DiffLineRanges = dict[str, set[int]]
"""Mapping of file path → set of commentable line numbers (head side)."""


def diff_line_ranges(
    repo: pygit2.Repository,
    base_ref: str,
    head_ref: str,
) -> DiffLineRanges:
    """Return the set of commentable line numbers per file in the diff.

    A line is "commentable" on GitHub if it appears in a diff hunk
    — either as a changed line or a context line on the HEAD side.

    The diff is computed from the **merge base** of the two refs,
    matching how GitHub computes the PR diff (three-dot diff).
    Without this, lines diverge when the base branch has received
    new commits since the PR was opened.

    Uses the ``new_lineno`` of each :class:`pygit2.DiffLine` to
    collect every line the GitHub review API will accept for
    ``side=RIGHT``.
    """
    base_commit = resolve_commit(repo, base_ref)
    head_commit = resolve_commit(repo, head_ref)

    # Use merge base to match GitHub's three-dot diff.
    merge_base_oid = repo.merge_base(base_commit.id, head_commit.id)
    if merge_base_oid is not None:
        mb_obj = repo.get(merge_base_oid)
        if mb_obj is not None:
            base_commit = mb_obj.peel(pygit2.Commit)

    d = repo.diff(base_commit, head_commit)

    ranges: DiffLineRanges = {}
    for patch in d:
        if patch is None:
            continue
        path = patch.delta.new_file.path
        lines: set[int] = set()
        for hunk in patch.hunks:
            for diff_line in hunk.lines:
                if diff_line.new_lineno >= 0:
                    lines.add(diff_line.new_lineno)
        if lines:
            ranges[path] = lines
    return ranges


def diff_line_ranges_left(
    repo: pygit2.Repository,
    base_ref: str,
    head_ref: str,
) -> DiffLineRanges:
    """Like :func:`diff_line_ranges` but for the BASE (LEFT) side.

    Returns commentable ``old_lineno`` values keyed by the **old**
    file path.  Used to validate ``side="LEFT"`` comments on
    deleted or modified lines.
    """
    base_commit = resolve_commit(repo, base_ref)
    head_commit = resolve_commit(repo, head_ref)

    merge_base_oid = repo.merge_base(base_commit.id, head_commit.id)
    if merge_base_oid is not None:
        mb_obj = repo.get(merge_base_oid)
        if mb_obj is not None:
            base_commit = mb_obj.peel(pygit2.Commit)

    d = repo.diff(base_commit, head_commit)

    ranges: DiffLineRanges = {}
    for patch in d:
        if patch is None:
            continue
        path = patch.delta.old_file.path
        lines: set[int] = set()
        for hunk in patch.hunks:
            for diff_line in hunk.lines:
                if diff_line.old_lineno >= 0:
                    lines.add(diff_line.old_lineno)
        if lines:
            ranges[path] = lines
    return ranges


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
