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
        1: make_chunk("r1_loader", name="load_alpha", path="alpha.py", blob="b_r1", repo_id=1),
        2: make_chunk("r2_loader", name="load_beta", path="beta.py", blob="b_r2", repo_id=2),
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
    resp = handle_search(SearchRequest(path=paths[1], query="load", scope=scenario.scope), store)
    by_id = {r.id: r for r in resp.results}

    expected_ids = {1: "r1_loader", 2: "r2_loader"}
    for repo_id, chunk_id in expected_ids.items():
        present = repo_id in scenario.expected_repos
        assert (chunk_id in by_id) is present
        if present and scenario.attributed:
            assert by_id[chunk_id].repo_path == paths[repo_id]
    if not scenario.attributed:
        assert all(r.repo_path is None for r in resp.results)


@parametrize_with_cases("scenario", cases=".cases_cross_repo")
def test_status_scope(
    scenario: ScopeScenario, two_repos: tuple[IndexStore, dict[int, str]]
) -> None:
    """Status reports refs for exactly the in-scope repos, attributed."""
    store, paths = two_repos
    resp = handle_status(StatusRequest(path=paths[1], scope=scenario.scope), store)

    expected_paths = {paths[r] for r in scenario.expected_repos}
    if scenario.attributed:
        assert {ref.repo_path for ref in resp.indexed_refs} == expected_paths
    else:
        assert len(resp.indexed_refs) == len(scenario.expected_repos)
        assert all(ref.repo_path is None for ref in resp.indexed_refs)
