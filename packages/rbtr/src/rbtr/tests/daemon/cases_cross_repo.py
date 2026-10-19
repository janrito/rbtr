"""Cases for searching and reporting status in one repo or in every repo.

Each case says how to search and how to ask for status, and which repos
the results should hold, so one search test and one status test cover
the single-repo and the every-repo requests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pytest_cases import case

from rbtr.daemon.handlers import (
    handle_search,
    handle_search_all,
    handle_status,
    handle_status_all,
)
from rbtr.daemon.messages import (
    SearchAllRequest,
    SearchRequest,
    SearchResponse,
    StatusAllRequest,
    StatusRequest,
    StatusResponse,
)
from rbtr.index.index import Index


@dataclass(frozen=True)
class ScopeScenario:
    """How to search and ask for status from a repo, and which repos the results hold."""

    search: Callable[[Index, str], SearchResponse]
    status: Callable[[Index, str], StatusResponse]
    # Repo indices (1, 2) expected in the result.
    expected_repos: frozenset[int]
    # Whether results should carry repo_path attribution.
    attributed: bool


@case(id="workspace")
def case_workspace() -> ScopeScenario:
    """Only the request's own repo, no search attribution."""
    return ScopeScenario(
        search=lambda index, path: handle_search(
            SearchRequest(repo_path=path, query="load"), index
        ),
        status=lambda index, path: handle_status(StatusRequest(repo_path=path), index),
        expected_repos=frozenset({1}),
        attributed=False,
    )


@case(id="all")
def case_all() -> ScopeScenario:
    """Every indexed repo, attributed by path."""
    return ScopeScenario(
        search=lambda index, _path: handle_search_all(SearchAllRequest(query="load"), index),
        status=lambda index, _path: handle_status_all(StatusAllRequest(), index),
        expected_repos=frozenset({1, 2}),
        attributed=True,
    )
