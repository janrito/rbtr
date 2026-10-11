"""What the daemon is doing when a read asks for a ref it has not indexed.

Each case names the build running, if any, and the refs watched in the
read's repo, and the error the read should get: "in progress" only when
the daemon will index that ref without being asked.
"""

from __future__ import annotations

from dataclasses import dataclass

from pytest_cases import case

from rbtr.daemon.messages import ActiveJob, ErrorCode, Scope


@dataclass(frozen=True)
class UnbuiltRefScenario:
    """The daemon's work when the read arrives, and the error it should get."""

    active_build: ActiveJob | None
    watched: list[str]
    expected: ErrorCode


@case(id="another_repo_building")
def case_unbuilt_another_repo_building(second_repo: str, unindexed_ref: str) -> UnbuiltRefScenario:
    """A build of the same commit in another repo indexes nothing here."""
    return UnbuiltRefScenario(
        active_build=ActiveJob(
            repo_path=second_repo,
            ref=unindexed_ref,
            phase="parsing",
            current=1,
            total=2,
            elapsed_seconds=0.0,
        ),
        watched=[],
        expected=ErrorCode.INDEX_NOT_BUILT,
    )


@case(id="another_commit_building")
def case_unbuilt_another_commit_building(fake_repo: str) -> UnbuiltRefScenario:
    """A build of another commit in the same repo does not index this one."""
    return UnbuiltRefScenario(
        active_build=ActiveJob(
            repo_path=fake_repo,
            ref="f" * 40,
            phase="parsing",
            current=1,
            total=2,
            elapsed_seconds=0.0,
        ),
        watched=[],
        expected=ErrorCode.INDEX_NOT_BUILT,
    )


@case(id="this_commit_building")
def case_unbuilt_this_commit_building(fake_repo: str, unindexed_ref: str) -> UnbuiltRefScenario:
    """The build running is the one the read waits for."""
    return UnbuiltRefScenario(
        active_build=ActiveJob(
            repo_path=fake_repo,
            ref=unindexed_ref,
            phase="parsing",
            current=1,
            total=2,
            elapsed_seconds=0.0,
        ),
        watched=[],
        expected=ErrorCode.INDEX_IN_PROGRESS,
    )


@case(id="watched_not_yet_built")
def case_unbuilt_watched_not_yet_built(unindexed_ref: str) -> UnbuiltRefScenario:
    """A watched ref is the daemon's to index, even before its build starts."""
    return UnbuiltRefScenario(
        active_build=None, watched=[unindexed_ref], expected=ErrorCode.INDEX_IN_PROGRESS
    )


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
