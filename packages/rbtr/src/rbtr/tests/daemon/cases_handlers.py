"""Where a status request is asked from, its scope, and the root its reply names."""

from __future__ import annotations

from dataclasses import dataclass

from pytest_cases import case

from rbtr.daemon.messages import Scope


@dataclass(frozen=True)
class StatusRootScenario:
    """Where a status request is asked from, its scope, and the root it should name."""

    repo_path: str
    scope: Scope
    expected: str | None


@case(id="from_the_root")
def case_status_from_the_root(fake_repo: str) -> StatusRootScenario:
    return StatusRootScenario(repo_path=fake_repo, scope=Scope.WORKSPACE, expected=fake_repo)


@case(id="from_a_subdirectory")
def case_status_from_a_subdirectory(fake_repo: str, fake_repo_subdir: str) -> StatusRootScenario:
    """A caller in a subdirectory learns the root its paths are relative to."""
    return StatusRootScenario(repo_path=fake_repo_subdir, scope=Scope.WORKSPACE, expected=fake_repo)


@case(id="unregistered")
def case_status_unregistered(second_repo: str) -> StatusRootScenario:
    """A repository with nothing indexed still has a root."""
    return StatusRootScenario(repo_path=second_repo, scope=Scope.WORKSPACE, expected=second_repo)


@case(id="scope_all")
def case_status_scope_all(fake_repo: str) -> StatusRootScenario:
    """Every repository's status names no single root."""
    return StatusRootScenario(repo_path=fake_repo, scope=Scope.ALL, expected=None)
