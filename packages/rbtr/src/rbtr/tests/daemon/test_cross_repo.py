"""Cross-repo search and status handlers.

Drives `handle_search` / `handle_status` directly against an
in-memory two-repo store — no socket round-trip — across the
workspace and all scopes (see `cases_cross_repo`).
"""

from __future__ import annotations

import pygit2
import pytest
from pytest_cases import parametrize_with_cases

from rbtr.daemon.dto import ResolvedRef
from rbtr.domain.models import RefSource
from rbtr.index.build import build_index
from rbtr.index.index import Index

from ..conftest import make_commit
from .cases_cross_repo import ScopeScenario


@pytest.fixture
def two_repos(index: Index, fake_repo: str, second_repo: str) -> tuple[Index, dict[int, str]]:
    """A store with two built repos and their `repo_id -> path` map.

    Repo 1 holds `load_alpha`, repo 2 `load_beta`; both names contain
    `load`, so a single query reaches both. Each is built at a commit
    that is now its HEAD.
    """
    alpha = b"""\
def load_alpha():
    pass
"""
    beta = b"""\
def load_beta():
    pass
"""
    paths = {1: fake_repo, 2: second_repo}
    for path, files in ((fake_repo, {"alpha.py": alpha}), (second_repo, {"beta.py": beta})):
        repo = pygit2.Repository(path)
        sha = str(make_commit(repo, files, parents=[repo.head.peel(pygit2.Commit).id]))
        repo.checkout_head(strategy=pygit2.GIT_CHECKOUT_FORCE)  # type: ignore[no-untyped-call]  # pygit2 untyped
        build_index(path, sha, index)
    return index, paths


@parametrize_with_cases("scenario", cases=".cases_cross_repo")
def test_search_scope(scenario: ScopeScenario, two_repos: tuple[Index, dict[int, str]]) -> None:
    """Search returns hits from exactly the in-scope repos, attributed."""
    index, paths = two_repos
    resp = scenario.search(index, paths[1])
    # The DTO carries no id; repo_path is the attribution signal. In
    # attributed (cross-repo) mode the set of repo_paths returned must be
    # exactly the in-scope repos; in workspace mode all hits are
    # unattributed and come from the queried repo only.
    if scenario.attributed:
        expected_paths = {paths[r] for r in scenario.expected_repos}
        assert {r.repo_path for r in resp.results} == expected_paths
        # Every repo at its latest indexed snapshot: no single one to name.
        assert resp.resolved is None
    else:
        assert resp.results
        assert all(r.repo_path is None for r in resp.results)
        head = str(pygit2.Repository(paths[1]).head.target)
        assert resp.resolved == ResolvedRef(sha=head, source=RefSource.HEAD)


@parametrize_with_cases("scenario", cases=".cases_cross_repo")
def test_status_scope(scenario: ScopeScenario, two_repos: tuple[Index, dict[int, str]]) -> None:
    """Status reports refs for exactly the in-scope repos, attributed."""
    index, paths = two_repos
    resp = scenario.status(index, paths[1])

    expected_paths = {paths[r] for r in scenario.expected_repos}
    assert {ref.repo_path for ref in resp.indexed_refs} == expected_paths
