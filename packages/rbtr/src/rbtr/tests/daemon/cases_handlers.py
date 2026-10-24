"""What the daemon is doing when a read asks for a ref it has not indexed.

Each case names the build running, if any, and the refs watched in the
read's repo, and the error the read should get: "in progress" only when
the daemon will index that ref without being asked.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygit2
from pytest_cases import case

from rbtr.daemon.messages import (
    ErrorCode,
    FindRefsRequest,
    IndexState,
    ListSymbolsRequest,
    ReadSymbolRequest,
    SearchRequest,
    StatusAllRequest,
    StatusRequest,
)
from rbtr.domain.models import SnapshotRef
from rbtr.index.index import Index
from rbtr.tests.conftest import make_commit


@dataclass(frozen=True)
class UnbuiltRefScenario:
    """The daemon's work when the read arrives, and the error it should get."""

    building: SnapshotRef | None
    watched: list[str]
    expected: ErrorCode


@case(id="another_repo_building")
def case_unbuilt_another_repo_building(
    second_repo: str, fake_repo: str, unindexed_ref: str, seeded_index: Index
) -> UnbuiltRefScenario:
    """A build of the same commit in another repo indexes nothing here."""
    other = pygit2.Repository(second_repo)
    other.remotes.create("source", fake_repo).fetch(["refs/heads/unindexed:refs/heads/building"])
    return UnbuiltRefScenario(
        building=seeded_index.register(second_repo).at(unindexed_ref),
        watched=[],
        expected=ErrorCode.INDEX_NOT_BUILT,
    )


@case(id="another_commit_building")
def case_unbuilt_another_commit_building(
    fake_repo: str, seeded_index: Index, unindexed_ref: str
) -> UnbuiltRefScenario:
    """A build of another commit in the same repo does not index this one."""
    repo = pygit2.Repository(fake_repo)
    sha = make_commit(repo, {"other.py": b"other = 1\n"}, ref="refs/heads/building")
    return UnbuiltRefScenario(
        building=seeded_index.repo(fake_repo).at(str(sha)),
        watched=[],
        expected=ErrorCode.INDEX_NOT_BUILT,
    )


@case(id="this_commit_building")
def case_unbuilt_this_commit_building(
    fake_repo: str, unindexed_ref: str, seeded_index: Index
) -> UnbuiltRefScenario:
    """The build running is the one the read waits for."""
    pygit2.Repository(fake_repo).references.create(
        "refs/heads/building", pygit2.Oid(hex=unindexed_ref)
    )
    return UnbuiltRefScenario(
        building=seeded_index.repo(fake_repo).at(unindexed_ref),
        watched=[],
        expected=ErrorCode.INDEX_IN_PROGRESS,
    )


@case(id="watched_not_yet_built")
def case_unbuilt_watched_not_yet_built(unindexed_ref: str) -> UnbuiltRefScenario:
    """A watched ref is the daemon's to index, even before its build starts."""
    return UnbuiltRefScenario(
        building=None, watched=[unindexed_ref], expected=ErrorCode.INDEX_IN_PROGRESS
    )


@dataclass(frozen=True)
class StatusRootScenario:
    """A status request, and the root its reply should name."""

    request: StatusRequest | StatusAllRequest
    expected: str | None


@case(id="from_the_root")
def case_status_from_the_root(fake_repo: str) -> StatusRootScenario:
    return StatusRootScenario(request=StatusRequest(repo_path=fake_repo), expected=fake_repo)


@case(id="from_a_subdirectory")
def case_status_from_a_subdirectory(fake_repo: str, fake_repo_subdir: str) -> StatusRootScenario:
    """A caller in a subdirectory learns the root its paths are relative to."""
    return StatusRootScenario(request=StatusRequest(repo_path=fake_repo_subdir), expected=fake_repo)


@case(id="unregistered")
def case_status_unregistered(second_repo: str) -> StatusRootScenario:
    """A repository with nothing indexed still has a root."""
    return StatusRootScenario(request=StatusRequest(repo_path=second_repo), expected=second_repo)


@case(id="every_repo")
def case_status_every_repo() -> StatusRootScenario:
    """Every repository's status names no single root."""
    return StatusRootScenario(request=StatusAllRequest(), expected=None)


@dataclass(frozen=True)
class IndexStateScenario:
    """A status request against an index, and the state its reply should report."""

    index: Index
    request: StatusRequest | StatusAllRequest
    expected: IndexState


# A case that needs more state than its fixtures makes it in its own body:
# a test fed by cases sets up every case's fixtures for each case, so a
# commit made by a fixture would move every case's HEAD.


@case(id="nothing_registered")
def case_state_nothing_registered(empty_index: Index, fake_repo: str) -> IndexStateScenario:
    return IndexStateScenario(empty_index, StatusRequest(repo_path=fake_repo), IndexState.NO_INDEX)


@case(id="nothing_registered_every_repo")
def case_state_nothing_registered_every_repo(empty_index: Index) -> IndexStateScenario:
    return IndexStateScenario(empty_index, StatusAllRequest(), IndexState.NO_INDEX)


@case(id="registered_never_built")
def case_state_registered_never_built(seeded_index: Index, second_repo: str) -> IndexStateScenario:
    seeded_index.register(second_repo)
    return IndexStateScenario(
        seeded_index, StatusRequest(repo_path=second_repo), IndexState.REPO_NOT_INDEXED
    )


@case(id="registered_never_built_every_repo")
def case_state_registered_never_built_every_repo(
    empty_index: Index, fake_repo: str
) -> IndexStateScenario:
    empty_index.register(fake_repo)
    return IndexStateScenario(empty_index, StatusAllRequest(), IndexState.REPO_NOT_INDEXED)


@case(id="head_past_the_index")
def case_state_head_past_the_index(seeded_index: Index, fake_repo: str) -> IndexStateScenario:
    """A commit after the indexed one: reads fall back to the indexed commit."""
    repo = pygit2.Repository(fake_repo)
    sig = pygit2.Signature("t", "t@t.t")
    head = repo.head.peel(pygit2.Commit)
    repo.create_commit("refs/heads/main", sig, sig, "next", head.tree.id, [head.id])
    return IndexStateScenario(
        seeded_index, StatusRequest(repo_path=fake_repo), IndexState.REF_NOT_INDEXED
    )


@case(id="head_indexed_every_repo")
def case_state_head_indexed_every_repo(seeded_index: Index) -> IndexStateScenario:
    return IndexStateScenario(seeded_index, StatusAllRequest(), IndexState.INDEXED)


# Each read a client can make at a ref: one the daemon has not indexed is
# an error, never an empty result.


@case(id="search")
def case_unindexed_read_search(fake_repo: str, unindexed_ref: str) -> SearchRequest:
    return SearchRequest(repo_path=fake_repo, query="load_config", ref=unindexed_ref)


@case(id="read_symbol")
def case_unindexed_read_read_symbol(fake_repo: str, unindexed_ref: str) -> ReadSymbolRequest:
    return ReadSymbolRequest(repo_path=fake_repo, symbol="load_config", ref=unindexed_ref)


@case(id="list_symbols")
def case_unindexed_read_list_symbols(fake_repo: str, unindexed_ref: str) -> ListSymbolsRequest:
    return ListSymbolsRequest(repo_path=fake_repo, file_path="src/config.py", ref=unindexed_ref)


@case(id="find_refs")
def case_unindexed_read_find_refs(fake_repo: str, unindexed_ref: str) -> FindRefsRequest:
    return FindRefsRequest(repo_path=fake_repo, symbol="load_config", ref=unindexed_ref)
