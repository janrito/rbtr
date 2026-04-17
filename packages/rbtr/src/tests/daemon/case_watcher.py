"""Test cases for ``rbtr.daemon.watcher.poll``.

Each case builds an ``IndexStore`` plus any git repos needed on
disk, then returns ``(store, expected_stale_heads)``.  The cases
are the only place watcher scenarios are described; the test in
``test_watcher.py`` asserts the single behaviour ``poll`` returns
the expected ``StaleHead`` list.

Cases are split by tag into two families:

``no_stale``
    ``poll`` returns ``[]``.
``stale``
    ``poll`` returns a non-empty list; the exact expected heads
    are part of the case output.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
from pytest_cases import case

from rbtr.daemon.watcher import StaleHead
from rbtr.index.store import IndexStore

_SIG = pygit2.Signature("t", "t@t.t")


def _init_repo(path: Path) -> pygit2.Repository:
    """Create a repo at *path* with one commit on refs/heads/main."""
    repo = pygit2.init_repository(str(path), bare=False)
    tb = repo.TreeBuilder()
    tb.insert("a.py", repo.create_blob(b"x = 1\n"), pygit2.GIT_FILEMODE_BLOB)
    repo.create_commit("refs/heads/main", _SIG, _SIG, "init", tb.write(), [])
    return repo


def _commit(repo: pygit2.Repository, name: str, content: bytes) -> str:
    """Append a second commit to *repo*; return its SHA."""
    tb = repo.TreeBuilder()
    tb.insert(name, repo.create_blob(content), pygit2.GIT_FILEMODE_BLOB)
    parents = [repo.head.target]
    new = repo.create_commit(
        "refs/heads/main", _SIG, _SIG, "c", tb.write(), parents
    )
    return str(new)


def _workdir(repo: pygit2.Repository) -> str:
    assert repo.workdir is not None
    return repo.workdir


# ── no_stale ─────────────────────────────────────────────────────────


@case(tags=["no_stale"])
def case_empty_store() -> tuple[IndexStore, list[StaleHead]]:
    """No repos registered at all."""
    return IndexStore(), []


@case(tags=["no_stale"])
def case_head_already_indexed(
    tmp_path: Path,
) -> tuple[IndexStore, list[StaleHead]]:
    """Registered repo whose HEAD is recorded in indexed_commits."""
    repo = _init_repo(tmp_path / "r")
    store = IndexStore()
    repo_id = store.register_repo(_workdir(repo))
    store.mark_indexed(repo_id, str(repo.head.target))
    return store, []


@case(tags=["no_stale"])
def case_registered_path_missing(
    tmp_path: Path,
) -> tuple[IndexStore, list[StaleHead]]:
    """Registered path is not a git repo: silently skipped."""
    store = IndexStore()
    store.register_repo(str(tmp_path / "does-not-exist"))
    return store, []


# ── stale ────────────────────────────────────────────────────────────


@case(tags=["stale"])
def case_head_never_indexed(
    tmp_path: Path,
) -> tuple[IndexStore, list[StaleHead]]:
    """First-time repo: registered but no indexed_commits row yet."""
    repo = _init_repo(tmp_path / "r")
    store = IndexStore()
    store.register_repo(_workdir(repo))
    return store, [StaleHead(_workdir(repo), str(repo.head.target))]


@case(tags=["stale"])
def case_new_commit_since_indexing(
    tmp_path: Path,
) -> tuple[IndexStore, list[StaleHead]]:
    """HEAD has advanced past the last indexed SHA."""
    repo = _init_repo(tmp_path / "r")
    store = IndexStore()
    repo_id = store.register_repo(_workdir(repo))
    store.mark_indexed(repo_id, str(repo.head.target))
    new_head = _commit(repo, "b.py", b"y = 2\n")
    return store, [StaleHead(_workdir(repo), new_head)]


@case(tags=["stale"])
def case_mixed_multi_repo(
    tmp_path: Path,
) -> tuple[IndexStore, list[StaleHead]]:
    """One repo up to date, one stale: only the stale one reports."""
    repo_fresh = _init_repo(tmp_path / "fresh")
    repo_stale = _init_repo(tmp_path / "stale")
    store = IndexStore()
    fresh_id = store.register_repo(_workdir(repo_fresh))
    store.register_repo(_workdir(repo_stale))
    store.mark_indexed(fresh_id, str(repo_fresh.head.target))
    return store, [
        StaleHead(_workdir(repo_stale), str(repo_stale.head.target))
    ]
