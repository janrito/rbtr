"""Behavioural test for ``rbtr.daemon.watcher.poll``.

One behaviour: ``poll(store)`` returns the set of repos whose
current HEAD is not recorded in ``indexed_commits``.  Scenarios
live in ``case_watcher.py``; three nested fixtures convert a
declarative scenario into real git repos on disk, a seeded
``IndexStore``, and the expected ``StaleHead`` list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygit2
from pytest_cases import fixture, parametrize_with_cases

from rbtr.daemon.watcher import StaleHead, poll
from rbtr.index.store import IndexStore
from tests.daemon.case_watcher import WatcherScenario


@dataclass(frozen=True)
class BuiltRepo:
    """A real git repo on disk plus the SHAs of its commits."""

    path: str
    shas: list[str]


@fixture
def sig() -> pygit2.Signature:
    """Author/committer identity shared by every built commit."""
    return pygit2.Signature("t", "t@t.t")


@fixture
@parametrize_with_cases("s", cases="tests.daemon.case_watcher")
def scenario(s: WatcherScenario) -> WatcherScenario:
    """Expose each case as the `scenario` fixture for downstream use."""
    return s


@fixture
def built_repos(
    scenario: WatcherScenario,
    sig: pygit2.Signature,
    tmp_path: Path,
) -> dict[str, BuiltRepo]:
    """Materialise every ``RepoSpec`` in *scenario* to disk."""
    out: dict[str, BuiltRepo] = {}
    for spec in scenario.repos:
        repo_dir = tmp_path / spec.name
        repo_dir.mkdir()
        shas: list[str] = []
        if spec.commits > 0:
            repo = pygit2.init_repository(str(repo_dir), bare=False)
            for i in range(spec.commits):
                tb = repo.TreeBuilder()
                tb.insert(
                    f"f{i}.py",
                    repo.create_blob(f"# commit {i}\n".encode()),
                    pygit2.GIT_FILEMODE_BLOB,
                )
                parents = [repo.head.target] if not repo.head_is_unborn else []
                new = repo.create_commit("refs/heads/main", sig, sig, f"c{i}", tb.write(), parents)
                shas.append(str(new))
        out[spec.name] = BuiltRepo(path=str(repo_dir), shas=shas)
    return out


@fixture
def seeded_store(
    scenario: WatcherScenario,
    built_repos: dict[str, BuiltRepo],
) -> IndexStore:
    """Register every repo in *scenario* and apply recorded marks."""
    store = IndexStore()
    for spec in scenario.repos:
        if not spec.register:
            continue
        repo_id = store.register_repo(built_repos[spec.name].path)
        marked_index = scenario.indexed_at.get(spec.name)
        if marked_index is not None:
            store.mark_indexed(repo_id, built_repos[spec.name].shas[marked_index])
    return store


@fixture
def expected_stale(
    scenario: WatcherScenario,
    built_repos: dict[str, BuiltRepo],
) -> list[StaleHead]:
    """Compute the expected ``poll`` output from the scenario."""
    return [
        StaleHead(
            repo_path=built_repos[name].path,
            new_ref=built_repos[name].shas[-1],
        )
        for name in scenario.expected_stale
    ]


def test_poll_reports_stale_heads(
    seeded_store: IndexStore,
    expected_stale: list[StaleHead],
) -> None:
    assert poll(seeded_store) == expected_stale
