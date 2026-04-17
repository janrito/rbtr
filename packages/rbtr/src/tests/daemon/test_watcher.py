"""Behavioural test for ``rbtr.daemon.watcher.poll``.

One behaviour: ``poll(store)`` returns the set of repos whose
current HEAD is not recorded in ``indexed_commits``.  Scenarios
live in ``case_watcher.py``; the shared fixture here converts a
declarative scenario into real git repos on disk and a real
``IndexStore``.
"""

from __future__ import annotations

from pathlib import Path

import pygit2
from pytest_cases import fixture, parametrize_with_cases

from rbtr.daemon.watcher import StaleHead, poll
from rbtr.index.store import IndexStore
from tests.daemon.case_watcher import RepoSpec, WatcherScenario


@fixture
@parametrize_with_cases("scenario", cases="tests.daemon.case_watcher")
def store_and_expected(
    scenario: WatcherScenario, tmp_path: Path
) -> tuple[IndexStore, list[StaleHead]]:
    """Build the scenario's repos and store, return (store, expected)."""
    store = IndexStore()
    shas_per_repo: dict[str, list[str]] = {}
    path_per_repo: dict[str, str] = {}

    for spec in scenario.repos:
        repo_dir = tmp_path / spec.name
        path = _create_repo(repo_dir, spec)
        path_per_repo[spec.name] = path
        shas_per_repo[spec.name] = _commit_history(repo_dir, spec)

        if spec.register:
            repo_id = store.register_repo(path)
            marked_index = scenario.indexed_at.get(spec.name)
            if marked_index is not None:
                store.mark_indexed(repo_id, shas_per_repo[spec.name][marked_index])

    expected = [
        StaleHead(
            repo_path=path_per_repo[name],
            new_ref=shas_per_repo[name][-1],
        )
        for name in scenario.expected_stale
    ]
    return store, expected


def test_poll_reports_stale_heads(
    store_and_expected: tuple[IndexStore, list[StaleHead]],
) -> None:
    store, expected = store_and_expected
    assert poll(store) == expected


# ── Scenario realisation ────────────────────────────────────────────
#
# These are not test helpers — they are the fixture's body, which
# converts a declarative ``WatcherScenario`` into concrete git repos
# on disk.  No test function calls them; they exist only to serve
# the ``store_and_expected`` fixture above.


_SIG = pygit2.Signature("t", "t@t.t")


def _create_repo(repo_dir: Path, spec: RepoSpec) -> str:
    """Make *repo_dir* look like the spec asks for; return its path."""
    repo_dir.mkdir()
    if spec.commits > 0:
        pygit2.init_repository(str(repo_dir), bare=False)
    return str(repo_dir)


def _commit_history(repo_dir: Path, spec: RepoSpec) -> list[str]:
    """Write *spec.commits* commits to *repo_dir*; return their SHAs."""
    if spec.commits == 0:
        return []
    repo = pygit2.Repository(str(repo_dir))
    shas: list[str] = []
    for i in range(spec.commits):
        tb = repo.TreeBuilder()
        tb.insert(
            f"f{i}.py",
            repo.create_blob(f"# commit {i}\n".encode()),
            pygit2.GIT_FILEMODE_BLOB,
        )
        parents = [repo.head.target] if not repo.head_is_unborn else []
        new = repo.create_commit(
            "refs/heads/main", _SIG, _SIG, f"c{i}", tb.write(), parents
        )
        shas.append(str(new))
    return shas
