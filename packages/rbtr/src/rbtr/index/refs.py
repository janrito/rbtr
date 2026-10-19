"""Which snapshot a read means: the dirty worktree, a resolved ref, or the latest indexed.

The rule spans git and the index, so it lives above both: it asks git
for the worktree's tree and for refs, and the index which snapshots are
indexed.
"""

from __future__ import annotations

from dataclasses import dataclass

from rbtr.config import config
from rbtr.domain.models import RefSource, Repo, SnapshotRef
from rbtr.errors import IndexNotBuiltError, MissingSnapshot, RbtrError
from rbtr.git import HEAD_REF, WORKTREE_REF, resolve_ref, worktree_tree_sha
from rbtr.index.index import Index


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedRead:
    """The snapshot a read resolved to, and how it was chosen."""

    at: SnapshotRef
    source: RefSource


def _indexed_worktree_ref(index: Index, repo_path: str, repo_id: int) -> SnapshotRef | None:
    """The worktree's tree SHA, when the tree is dirty and indexed.

    `None` when the worktree matches HEAD's tree, when the repo
    is gone, or when that tree has never been indexed — in each
    case the caller wants a commit instead.
    """
    tree_sha = worktree_tree_sha(repo_path, max_untracked_files=config.max_untracked_files)
    if tree_sha is None:
        return None
    at = SnapshotRef(repo_id=repo_id, snapshot_sha=tree_sha)
    return at if index.has_indexed(at=at) else None


def _latest_indexed_ref(index: Index, repo_id: int) -> SnapshotRef | None:
    """The most recently indexed snapshot for one repo; `None` if it has none."""
    indexed = index.list_indexed_snapshots(repo_id)
    return SnapshotRef(repo_id=repo_id, snapshot_sha=indexed[0][0]) if indexed else None


def latest_ref(index: Index, repo: Repo) -> SnapshotRef | None:
    """Resolve the most recent indexed ref for one repo.

    Prefers the current worktree tree SHA when the worktree is
    dirty and that tree has been indexed; otherwise falls back
    to the newest indexed commit.  Returns `None` when the repo
    has no indexed commits at all.
    """
    dirty = _indexed_worktree_ref(index, repo.repo_path, repo.repo_id)
    return dirty if dirty is not None else _latest_indexed_ref(index, repo.repo_id)


def list_latest_refs(index: Index) -> list[SnapshotRef]:
    """Return one `SnapshotRef` per registered repo with indexed data.

    Applies `latest_ref` to every repo from `list_repos`,
    skipping repos that have never been indexed.  The basis for
    cross-repo search: the returned list is passed straight to
    `search()`.
    """
    refs: list[SnapshotRef] = []
    for repo in index.list_repos():
        ref = latest_ref(index, repo)
        if ref is not None:
            refs.append(ref)
    return refs


def resolve_read_ref(
    index: Index,
    repo_path: str,
    repo_id: int,
    requested_ref: str | None,
    *,
    require_indexed: bool = False,
) -> ResolvedRead:
    """Resolve a ref for read operations to the snapshot it names.

    When *requested_ref* is `None`, prefers the worktree's tree SHA
    when the tree is dirty and indexed (`_indexed_worktree_ref`).
    Otherwise resolves `"HEAD"`.  Explicit `"HEAD"` always resolves
    to the committed state.  Falls back to the latest indexed commit
    when the repo is missing.  Raises `RbtrError` if the ref cannot
    be resolved.

    When *require_indexed* is set, the resolved SHA must be usable:

    - An explicit *requested_ref* whose SHA is not indexed is an
      error (`ensure_indexed`) rather than a silent empty result.
    - The implicit worktree/HEAD path falls back to the latest
      indexed commit when the resolved SHA is not indexed (e.g. a
      build is still finalising), and only errors when the repo has
      no indexed commits at all. So an older indexed version of a
      symbol is preferred over an error.

    The result says which of these choices was made (`RefSource`).
    """

    explicit = requested_ref is not None
    if requested_ref is None:
        if (dirty := _indexed_worktree_ref(index, repo_path, repo_id)) is not None:
            return ResolvedRead(at=dirty, source=RefSource.WORKTREE)
        requested_ref = HEAD_REF
    try:
        sha = resolve_ref(repo_path, requested_ref, max_untracked_files=config.max_untracked_files)
    except RbtrError:
        if (
            requested_ref == HEAD_REF
            and (latest := _latest_indexed_ref(index, repo_id)) is not None
        ):
            return ResolvedRead(at=latest, source=RefSource.LATEST_INDEXED)
        msg = f"Cannot resolve ref '{requested_ref}' in {repo_path}"
        raise RbtrError(msg) from None
    at = SnapshotRef(repo_id=repo_id, snapshot_sha=sha)
    if require_indexed and not index.has_indexed(at=at):
        if not explicit and (latest := _latest_indexed_ref(index, repo_id)) is not None:
            return ResolvedRead(at=latest, source=RefSource.LATEST_INDEXED)
        ensure_indexed(index, at, requested_ref)
    return ResolvedRead(at=at, source=RefSource.REQUESTED if explicit else RefSource.HEAD)


def ensure_indexed(index: Index, at: SnapshotRef, requested_ref: str) -> None:
    """Raise `IndexNotBuiltError` if the snapshot at *at* is not indexed.

    *requested_ref* is what the client asked for, which the message
    quotes; *at* is what it resolved to.

    The daemon's `_dispatch` upgrades this to an "index is building"
    message when it will index that snapshot; inline callers see the
    plain "not indexed" guidance.
    """
    if index.has_indexed(at=at):
        return
    if requested_ref == WORKTREE_REF:
        msg = "Working tree is not indexed yet — run rbtr watch first"
    else:
        msg = f"Ref '{requested_ref}' is not indexed — run rbtr watch first"
    missing = MissingSnapshot(repo_id=at.repo_id, snapshot_sha=at.snapshot_sha, ref=requested_ref)
    raise IndexNotBuiltError(msg, missing=missing)
