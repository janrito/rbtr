"""Which snapshots this eval run measures.

The corpus is each clone's **HEAD**: `dvc.yaml` indexes the repos
without naming a ref, so `rbtr watch` builds `resolve_ref(path,
"HEAD")`, and the `index` stage then runs `gc --keep-head-only` per
repo. One indexed snapshot per repo, and it is HEAD, is therefore an
invariant this pipeline maintains rather than a hope about the
database.

`corpus_refs` states that rule and refuses to answer when it does not
hold. It is the only thing that builds a `Corpus`, so a report taking
one reports over a set of snapshots that has been checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rbtr.domain.models import SnapshotRef
from rbtr.errors import RbtrError
from rbtr.git import head_sha
from rbtr.index.index import Index


@dataclass(frozen=True, slots=True)
class Corpus:
    """The snapshots an eval run measures: one per repo, each at HEAD.

    The report queries join these refs, so every count covers this set
    and nothing else.
    """

    refs: tuple[SnapshotRef, ...]


def corpus_refs(index: Index) -> Corpus:
    """Every registered repo at HEAD, in registration order.

    Raises `RbtrError` when a repo's HEAD is not indexed, or when a
    repo holds an indexed snapshot that is not its HEAD — the latter
    doubles every per-snapshot count in the reports.
    """
    refs: list[SnapshotRef] = []
    for repo in index.repos():
        head = head_sha(repo.repo_path)
        indexed = {s.snapshot_sha for s in index.indexed_snapshots(repo)}
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
        refs.append(repo.at(head))
    return Corpus(refs=tuple(refs))


def corpus_ref(index: Index, slug: str) -> SnapshotRef:
    """The corpus snapshot for the repo whose directory is named *slug*.

    Raises `RbtrError` when no registered repo matches, or when the
    index fails the checks `corpus_refs` makes.
    """
    for ref in corpus_refs(index).refs:
        if Path(ref.repo.repo_path).name == slug:
            return ref
    msg = f"No indexed repo named {slug!r}. Run the `index` stage first."
    raise RbtrError(msg)
