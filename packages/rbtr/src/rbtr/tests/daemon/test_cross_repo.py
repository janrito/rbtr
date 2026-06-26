"""Cross-repo search and status handlers.

Drives `handle_search` / `handle_status` directly against an
in-memory two-repo store — no socket round-trip — across the
workspace and all scopes (see `cases_cross_repo`).
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.daemon.handlers import handle_search, handle_status
from rbtr.daemon.messages import SearchRequest, StatusRequest
from rbtr.index.models import Snapshot
from rbtr.index.store import IndexStore

from ..index.conftest import make_chunk
from .cases_cross_repo import ScopeScenario


@pytest.fixture
def two_repos() -> Generator[tuple[IndexStore, dict[int, str]]]:
    """A store with two indexed repos and their `repo_id -> path` map.

    Repo 1 holds chunk `r1_loader`, repo 2 holds `r2_loader`; both
    names contain `load` so a single query reaches both.
    """
    paths = {1: "/cross/repo_one", 2: "/cross/repo_two"}
    chunks = {
        1: make_chunk("r1_loader", name="load_alpha", path="alpha.py", blob="b_r1"),
        2: make_chunk("r2_loader", name="load_beta", path="beta.py", blob="b_r2"),
    }
    store = IndexStore(writable=True)
    for repo_id, chunk in chunks.items():
        with store.session() as ws:
            ws.register_repo(paths[repo_id])
            ws.add_chunk(chunk)
            ws.insert_snapshots(
                [Snapshot(commit_sha="head", file_path=chunk.file_path, blob_sha=chunk.blob_sha)],
                repo_id=repo_id,
            )
            ws.mark_indexed(repo_id, "head")
    yield store, paths
    store.close()


@parametrize_with_cases("scenario", cases=".cases_cross_repo")
def test_search_scope(
    scenario: ScopeScenario, two_repos: tuple[IndexStore, dict[int, str]]
) -> None:
    """Search returns hits from exactly the in-scope repos, attributed."""
    store, paths = two_repos
    resp = handle_search(
        SearchRequest(repo_path=paths[1], query="load", scope=scenario.scope), store
    )
    # The DTO carries no id; repo_path is the attribution signal. In
    # attributed (cross-repo) mode the set of repo_paths returned must be
    # exactly the in-scope repos; in workspace mode all hits are
    # unattributed and come from the queried repo only.
    if scenario.attributed:
        expected_paths = {paths[r] for r in scenario.expected_repos}
        assert {r.repo_path for r in resp.results} == expected_paths
    else:
        assert resp.results
        assert all(r.repo_path is None for r in resp.results)


@parametrize_with_cases("scenario", cases=".cases_cross_repo")
def test_status_scope(
    scenario: ScopeScenario, two_repos: tuple[IndexStore, dict[int, str]]
) -> None:
    """Status reports refs for exactly the in-scope repos, attributed."""
    store, paths = two_repos
    resp = handle_status(StatusRequest(repo_path=paths[1], scope=scenario.scope), store)

    expected_paths = {paths[r] for r in scenario.expected_repos}
    assert {ref.repo_path for ref in resp.indexed_refs} == expected_paths
