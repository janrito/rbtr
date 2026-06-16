"""Git operations for the code index.

Every public function takes a `repo_path: str` and opens the
repository internally — callers never handle `pygit2` types
directly.  The only exception is `read_head`, which catches
all errors and returns `None` for tolerance in the watcher's
poll loop.

Public surface
--------------

- `WORKTREE_REF`         — symbolic ref naming the working tree
- `normalise_repo_path` — resolve any path to its git root
- `FileEntry`           — dataclass for indexable files
- `is_binary`           — null-byte heuristic
- `read_head`           — tolerant HEAD read for polling
- `worktree_tree_sha`   — working-tree identity (tree SHA or None)
- `filter_tree_shas`    — batch tree-type check
- `resolve_ref`         — unified ref → SHA resolution
- `list_files`          — file listing (git tree or working tree)
- `read_blob`           — single blob read
- `changed_files`       — diff between two refs (or ref vs worktree)
- `names_for_commits`   — SHA → symbolic name mapping
- `head_sha`            — current HEAD SHA (strict)
- `local_ref_shas`      — SHAs for local branches/tags
- `resolve_refs_to_shas` — bulk ref resolution
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pathspec
import pygit2
import structlog

from rbtr.errors import RbtrError

# Symbolic ref naming the current working tree; `resolve_ref` maps it
# to the working-tree tree SHA.
WORKTREE_REF = "worktree"

if TYPE_CHECKING:
    from collections.abc import Iterator


log = structlog.get_logger(__name__)

_HEX_SHA_LEN = 40


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


# ── Internal helpers ─────────────────────────────────────────────────


def _open_repo(path: str) -> pygit2.Repository:
    """Open the git repository containing *path*.

    Walks upward from *path* to find the `.git` directory.
    Raises `RbtrError` if *path* is not inside a git repository.
    """
    discovered = pygit2.discover_repository(path)
    if discovered is None:
        msg = f"Not a git repository: {path}"
        raise RbtrError(msg)
    return pygit2.Repository(discovered)


def _resolve_commit(repo: pygit2.Repository, ref: str) -> pygit2.Commit:
    """Resolve *ref* to a `pygit2.Commit`.

    Accepts a SHA, branch name, or tag. Tries the bare ref first,
    then `origin/<ref>` for remote-tracking branches.

    Raises `RbtrError` if neither resolves.
    """
    for candidate in (ref, f"origin/{ref}"):
        try:
            obj = repo.revparse_single(candidate)
            return obj.peel(pygit2.Commit)
        except (KeyError, ValueError):
            continue
    msg = f"Cannot resolve ref '{ref}' (tried local and origin remote)"
    raise RbtrError(msg)


def _looks_like_sha(ref: str) -> bool:
    """True if *ref* is a full 40-char hex SHA."""
    return len(ref) == _HEX_SHA_LEN and all(c in "0123456789abcdef" for c in ref.lower())


def _short_ref_name(ref_name: str) -> str:
    """Strip `refs/heads/` and `refs/tags/` from a full ref name.

    `refs/remotes/<n>` keeps the `<remote>/<branch>` portion so it
    stays unambiguous against local branches.
    """
    for prefix in ("refs/heads/", "refs/tags/"):
        if ref_name.startswith(prefix):
            return ref_name[len(prefix) :]
    if ref_name.startswith("refs/remotes/"):
        return ref_name[len("refs/remotes/") :]
    return ref_name


def _walk_tree(
    repo: pygit2.Repository,
    tree: pygit2.Tree,
    prefix: str,
) -> Iterator[tuple[str, pygit2.Blob]]:
    """Recursively walk *tree*, yielding `(path, blob)` pairs."""
    for entry in tree:
        path = f"{prefix}{entry.name}" if not prefix else f"{prefix}/{entry.name}"
        match entry.type_str:
            case "tree":
                obj = repo.get(entry.id)
                if isinstance(obj, pygit2.Tree):
                    yield from _walk_tree(repo, obj, path)
            case "blob":
                obj = repo.get(entry.id)
                if isinstance(obj, pygit2.Blob):
                    yield path, obj


def _diff_paths(diff: pygit2.Diff) -> set[str]:
    """Extract changed file paths from a pygit2 diff."""
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


# ── Path filtering ───────────────────────────────────────────────────


def is_binary(data: bytes, sample_size: int = 8192) -> bool:
    """Heuristic binary detection.

    Returns `True` if a null byte appears in the first
    *sample_size* bytes of *data*.
    """
    return b"\x00" in data[:sample_size]


# ── Public API ───────────────────────────────────────────────────────


def normalise_repo_path(path: str) -> str:
    """Resolve *path* to the git repository root.

    Accepts any path inside a repo (including subdirectories).
    Returns the canonical workdir path without trailing slash.
    """
    discovered = pygit2.discover_repository(str(Path(path).resolve()))
    if discovered is None:
        msg = f"Not a git repository: {path}"
        raise RbtrError(msg)
    repo = pygit2.Repository(discovered)
    workdir = repo.workdir
    if workdir is None:
        msg = f"Bare repository not supported: {path}"
        raise RbtrError(msg)
    return workdir.rstrip("/")


def read_head(repo_path: str) -> str | None:
    """Return the current HEAD SHA for the repo at *repo_path*.

    Returns `None` if the repo is unborn, missing, or unreadable.
    Does not raise — intended for pollers that should tolerate
    misconfigured repos.
    """
    try:
        repo = pygit2.Repository(repo_path)
        if repo.head_is_unborn:
            return None
        return str(repo.head.target)
    except pygit2.GitError:
        return None


def worktree_tree_sha(repo_path: str) -> str | None:
    """Return a tree SHA representing the current working-tree state.

    Computes a git tree object that includes all committed files
    plus any working-tree modifications (staged, unstaged, and
    untracked).  The tree SHA is content-addressed: same file
    contents produce the same SHA, and any edit changes it.

    Returns `None` when the working tree is clean (the tree SHA
    would equal HEAD's tree SHA) or when the repo is unreadable.

    Side effects: writes tree and blob objects to `.git/objects`
    when the working tree is dirty.  These are loose objects
    pruned by normal `git gc`.  The on-disk staging area is not
    modified (`index.read()` resets the in-memory index after
    `write_tree`).  A clean working tree returns `None` without
    writing any objects.
    """
    try:
        repo = _open_repo(repo_path)
        if not repo.status(untracked_files="normal", ignored=False):
            return None
        if repo.head_is_unborn:
            return None
        head_tree = str(repo.head.peel(pygit2.Tree).id)
        repo.index.read()
        repo.index.add_all()
        tree_sha = str(repo.index.write_tree())
        repo.index.read()  # reset in-memory index; on-disk staging untouched
    except (RbtrError, pygit2.GitError):
        return None
    else:
        return None if tree_sha == head_tree else tree_sha


def _is_tree_sha(repo: pygit2.Repository, sha: str) -> bool:
    """Return `True` if *sha* is a tree object in *repo*."""
    try:
        obj = repo.get(sha)
    except (ValueError, pygit2.GitError):
        return False
    else:
        return obj is not None and obj.type == pygit2.GIT_OBJECT_TREE


def filter_tree_shas(repo_path: str, shas: list[str]) -> list[str]:
    """Return the subset of *shas* that are tree objects in the repo.

    Opens the repo once and checks each SHA.  Returns an empty
    list if the repo is missing or unreadable.
    """
    try:
        repo = _open_repo(repo_path)
    except RbtrError:
        return []
    return [sha for sha in shas if _is_tree_sha(repo, sha)]


def resolve_ref(repo_path: str, ref: str) -> str:
    """Resolve *ref* to a commit or tree SHA.

    Handles several ref kinds:

    - `"worktree"` — computes the working-tree tree SHA via
      `worktree_tree_sha`.  Raises `RbtrError` when clean.
    - Raw 40-char hex SHA — returned as-is.  This preserves
      backward compatibility with callers that pass stored
      SHAs (commit or tree) without needing repo access.
    - Symbolic refs (`"HEAD"`, `"main"`, etc.) — resolved via
      `_resolve_commit`.
    """
    if ref == WORKTREE_REF:
        sha = worktree_tree_sha(repo_path)
        if sha is None:
            msg = "Working tree is clean — no worktree ref to resolve"
            raise RbtrError(msg)
        return sha
    if _looks_like_sha(ref):
        return ref
    repo = _open_repo(repo_path)
    return str(_resolve_commit(repo, ref).id)


def names_for_commits(repo_path: str, shas: list[str]) -> dict[str, list[str]]:
    """Return a map from SHA to the symbolic names that point to it.

    Names include `"HEAD"` when HEAD resolves to the commit, plus short
    forms of every matching local branch and tag.  Remote-tracking branches
    are kept as `origin/<n>`.  Worktree tree SHAs (matching the current
    `worktree_tree_sha`) are labelled `"working tree"`.  SHAs with no
    matching ref get an empty list.
    """
    names: dict[str, list[str]] = {sha: [] for sha in shas}
    wanted = set(names)
    if not wanted:
        return names

    try:
        repo = _open_repo(repo_path)
    except RbtrError:
        return names

    try:
        if not repo.head_is_unborn:
            current_head = str(repo.head.target)
            if current_head in wanted:
                names[current_head].append("HEAD")
    except pygit2.GitError:
        pass

    # Label the current worktree tree SHA.
    wt_sha = worktree_tree_sha(repo_path)
    if wt_sha is not None and wt_sha in wanted:
        names[wt_sha].append("working tree")

    for ref_name in repo.references:
        try:
            target = repo.references[ref_name].peel(pygit2.Commit)
        except (KeyError, pygit2.GitError):
            continue
        sha = str(target.id)
        if sha not in wanted:
            continue
        short = _short_ref_name(ref_name)
        if short not in names[sha]:
            names[sha].append(short)
    return names


def head_sha(repo_path: str) -> str:
    """Return the current HEAD SHA.

    Raises `RbtrError` if HEAD is unborn (no commits yet).
    """
    repo = _open_repo(repo_path)
    if repo.head_is_unborn:
        msg = "Cannot read HEAD: repository has no commits"
        raise RbtrError(msg)
    return str(repo.head.target)


def local_ref_shas(
    repo_path: str,
    prefixes: tuple[str, ...] = ("refs/heads/", "refs/tags/"),
) -> set[str]:
    """Return commit SHAs for local refs matching *prefixes*.

    Iterates over all references and collects those whose name
    starts with one of *prefixes*.  Malformed or un-peelable refs
    are silently skipped.
    """
    repo = _open_repo(repo_path)
    out: set[str] = set()
    for name in repo.references:
        if not name.startswith(prefixes):
            continue
        try:
            out.add(str(repo.references[name].peel(pygit2.Commit).id))
        except (pygit2.GitError, KeyError):
            log.debug("skipping_malformed_ref", ref=name, exc_info=True)
            continue
    return out


def resolve_refs_to_shas(repo_path: str, refs: list[str]) -> set[str]:
    """Bulk-resolve symbolic refs to commit SHAs.

    Raises `RbtrError` if any ref cannot be resolved.
    """
    repo = _open_repo(repo_path)
    return {str(_resolve_commit(repo, ref).id) for ref in refs}


def _resolve_tree(repo: pygit2.Repository, ref: str) -> pygit2.Tree:
    """Resolve *ref* to a `pygit2.Tree`.

    If *ref* is already a tree SHA, returns the tree directly.
    Otherwise resolves via `_resolve_commit` and returns the
    commit's tree.
    """
    if _looks_like_sha(ref) and _is_tree_sha(repo, ref):
        obj = repo.get(ref)
        if not isinstance(obj, pygit2.Tree):  # pragma: no cover
            msg = f"Expected tree object for {ref}"
            raise RbtrError(msg)
        return obj
    return _resolve_commit(repo, ref).tree


def list_files(
    repo_path: str,
    ref: str,
    *,
    max_file_size: int,
    ignore: pathspec.PathSpec | None = None,
) -> Iterator[FileEntry]:
    """Yield every indexable file at *ref*.

    *ref* may be a commit SHA, branch name, or a tree SHA
    (e.g. from `worktree_tree_sha`).  Tree SHAs are walked
    directly; commit refs are peeled to their tree first.

    Applies layered filtering:

    1. `.gitignore` rules via `repo.path_is_ignored`.
    2. `.rbtrignore` rules via *ignore* (when provided).
    3. Files larger than *max_file_size* are skipped.
    4. Binary files (null-byte heuristic) are skipped.
    """
    repo = _open_repo(repo_path)
    tree = _resolve_tree(repo, ref)

    for path, blob in _walk_tree(repo, tree, ""):
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
    repo_path: str,
    ref: str,
    path: str,
) -> pygit2.Blob | None:
    """Read a single blob from the object store.

    Returns the `pygit2.Blob` for *path* at *ref*, or `None` if
    the file doesn't exist at that ref.
    """
    try:
        repo = _open_repo(repo_path)
    except RbtrError:
        return None
    try:
        commit = _resolve_commit(repo, ref)
    except RbtrError:
        return None
    try:
        entry = commit.tree[path]
    except KeyError:
        return None
    obj = repo.get(entry.id)
    if isinstance(obj, pygit2.Blob):
        return obj
    return None


def changed_files(
    repo_path: str,
    base_ref: str,
    head_ref: str,
) -> set[str]:
    """Return file paths that differ between *base_ref* and *head_ref*.

    Both refs may be commit SHAs, branch names, or tree SHAs
    (e.g. from `worktree_tree_sha`).  Tree-to-tree, tree-to-commit,
    and commit-to-commit diffs are all supported.

    Includes added, modified, and deleted files.
    """
    if base_ref == head_ref:
        return set()

    repo = _open_repo(repo_path)
    base_tree = _resolve_tree(repo, base_ref)
    head_tree = _resolve_tree(repo, head_ref)
    # pygit2 supports tree-to-tree diff at runtime but the type
    # stubs only declare Commit|Oid|Reference overloads.
    diff = repo.diff(base_tree, head_tree)  # type: ignore[call-overload]  # pygit2 stubs
    return _diff_paths(diff)
