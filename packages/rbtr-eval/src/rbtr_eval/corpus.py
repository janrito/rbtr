"""Which snapshots this eval run measures.

The corpus is each clone's **HEAD**: `dvc.yaml` indexes the repos
without naming a ref, so `rbtr index` builds `resolve_ref(path,
"HEAD")`, and the `index` stage then runs `gc --keep-head-only` per
repo. One indexed snapshot per repo, and it is HEAD, is therefore an
invariant this pipeline maintains rather than a hope about the
database.

`corpus_refs` states that invariant and refuses to answer when it
does not hold, which is what entitles the report queries to join
`indexed_snapshots` directly instead of binding a snapshot set.
"""

from __future__ import annotations

from pathlib import Path

from rbtr.domain.models import SnapshotRef
from rbtr.errors import RbtrError
from rbtr.git import head_sha
from rbtr.index.store import IndexStore


def corpus_refs(store: IndexStore) -> list[SnapshotRef]:
    """Every registered repo at HEAD, in registration order.

    Raises `RbtrError` when a repo's HEAD is not indexed, or when a
    repo holds an indexed snapshot that is not its HEAD — the latter
    doubles every per-snapshot count in the reports.
    """
    refs: list[SnapshotRef] = []
    for repo in store.list_repos():
        head = head_sha(repo.repo_path)
        indexed = {sha for sha, _ in store.list_indexed_snapshots(repo.repo_id)}
        if head not in indexed:
            msg = (
                f"{repo.repo_path}: HEAD {head} is not indexed, so the eval "
                f"cannot describe this repo. Run the `index` stage."
            )
            raise RbtrError(msg)
        if extra := sorted(indexed - {head}):
            msg = (
                f"{repo.repo_path}: indexed snapshots besides HEAD {head}: "
                f"{', '.join(extra)}. Counts over this repo would include "
                f"every one of them. A dirty worktree indexed by the daemon "
                f"is the usual cause; commit or clean it, then run "
                f"`rbtr gc --keep-head-only --repo-path {repo.repo_path}`."
            )
            raise RbtrError(msg)
        refs.append(SnapshotRef(repo_id=repo.repo_id, snapshot_sha=head))
    return refs


def corpus_ref(store: IndexStore, slug: str) -> SnapshotRef:
    """The corpus snapshot for the repo whose directory is named *slug*.

    Raises `RbtrError` when no registered repo matches, or when the
    index fails the checks `corpus_refs` makes.
    """
    paths = {repo.repo_id: repo.repo_path for repo in store.list_repos()}
    for ref in corpus_refs(store):
        if Path(paths[ref.repo_id]).name == slug:
            return ref
    msg = f"No indexed repo named {slug!r}. Run the `index` stage first."
    raise RbtrError(msg)
