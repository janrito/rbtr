"""Scope cases for cross-repo search and status handlers.

Each case names a `Scope` and the repos expected in the result,
so one search test and one status test cover both the workspace
(single-repo) and all (cross-repo) branches.
"""

from __future__ import annotations

from dataclasses import dataclass

from pytest_cases import case

from rbtr.daemon.messages import Scope


@dataclass(frozen=True)
class ScopeScenario:
    """A scope plus the repo indices its result should contain."""

    scope: Scope
    # Repo indices (1, 2) expected in the result.
    expected_repos: frozenset[int]
    # Whether results should carry repo_path attribution.
    attributed: bool


@case(id="workspace")
def case_workspace() -> ScopeScenario:
    """Default scope: only the request's own repo, no attribution."""
    return ScopeScenario(scope=Scope.WORKSPACE, expected_repos=frozenset({1}), attributed=False)


@case(id="all")
def case_all() -> ScopeScenario:
    """Cross-repo scope: every indexed repo, attributed by path."""
    return ScopeScenario(scope=Scope.ALL, expected_repos=frozenset({1, 2}), attributed=True)
