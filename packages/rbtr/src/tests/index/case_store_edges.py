"""Scenarios for edge storage and retrieval.

Each case returns an ``EdgeScenario`` describing which edges the
store should contain (across which commits / repos) and what
every public read method should return for a family of queries.

A shared fixture in ``test_store_edges.py`` turns the scenario
into a real ``IndexStore`` and yields ``(store, scenario)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Edge, EdgeKind


@dataclass(frozen=True)
class RepoEdges:
    """Edges scoped to a single (repo, commit) pair.

    The fixture inserts every list under ``per_commit`` with its
    own ``insert_edges`` call; tests read back with the matching
    commit.
    """

    # {commit_sha: [edges]}
    per_commit: dict[str, list[Edge]] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeScenario:
    """Declarative edge-family test data."""

    repo_paths: list[str] = field(default_factory=lambda: ["/default"])
    per_repo: list[RepoEdges] = field(default_factory=list)

    # ``(repo_id, commit)`` \u2192 expected edges visible without filter.
    expected_edges: dict[tuple[int, str], list[Edge]] = field(
        default_factory=dict
    )
    # ``(repo_id, commit, source_id)`` \u2192 expected edges with ``source_id`` filter.
    expected_by_source: dict[tuple[int, str, str], list[Edge]] = field(
        default_factory=dict
    )
    # ``(repo_id, commit, EdgeKind)`` \u2192 expected edges with ``kind`` filter.
    expected_by_kind: dict[tuple[int, str, EdgeKind], list[Edge]] = field(
        default_factory=dict
    )


# ── Shared edge instances ────────────────────────────────────────────
#
# Edges are tiny structural records \u2014 no need for a shared cases_common
# style file.  Named here for case readability.

_MATH_CALLS_CLASS = Edge(
    source_id="math_1", target_id="math_class_1", kind=EdgeKind.CALLS
)
_A_CALLS_B = Edge(source_id="a", target_id="b", kind=EdgeKind.CALLS)
_C_IMPORTS_D = Edge(source_id="c", target_id="d", kind=EdgeKind.IMPORTS)


# ── Cases ────────────────────────────────────────────────────────────


def case_empty_store() -> EdgeScenario:
    """No repos, no edges."""
    return EdgeScenario(repo_paths=[])


def case_single_edge_single_commit() -> EdgeScenario:
    """One edge on one commit; other commits return nothing."""
    return EdgeScenario(
        repo_paths=["/r"],
        per_repo=[RepoEdges(per_commit={"head": [_MATH_CALLS_CLASS]})],
        expected_edges={
            (1, "head"): [_MATH_CALLS_CLASS],
            (1, "other"): [],
        },
    )


def case_two_edges_mixed_kinds() -> EdgeScenario:
    """Two edges of different kinds; filters return each separately."""
    edges = [_A_CALLS_B, _C_IMPORTS_D]
    return EdgeScenario(
        repo_paths=["/r"],
        per_repo=[RepoEdges(per_commit={"head": list(edges)})],
        expected_edges={(1, "head"): list(edges)},
        expected_by_source={
            (1, "head", "a"): [_A_CALLS_B],
            (1, "head", "c"): [_C_IMPORTS_D],
            (1, "head", "nonexistent"): [],
        },
        expected_by_kind={
            (1, "head", EdgeKind.CALLS): [_A_CALLS_B],
            (1, "head", EdgeKind.IMPORTS): [_C_IMPORTS_D],
        },
    )


def case_same_edge_on_two_commits() -> EdgeScenario:
    """Edges are commit-scoped: identical edge duplicated per commit."""
    return EdgeScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoEdges(
                per_commit={
                    "a": [_A_CALLS_B],
                    "b": [_A_CALLS_B],
                }
            )
        ],
        expected_edges={
            (1, "a"): [_A_CALLS_B],
            (1, "b"): [_A_CALLS_B],
        },
    )


def case_two_repos_isolated() -> EdgeScenario:
    """Same edge on same commit in two repos; reads are scoped."""
    return EdgeScenario(
        repo_paths=["/r1", "/r2"],
        per_repo=[
            RepoEdges(per_commit={"head": [_A_CALLS_B]}),
            RepoEdges(per_commit={"head": [_C_IMPORTS_D]}),
        ],
        expected_edges={
            (1, "head"): [_A_CALLS_B],
            (2, "head"): [_C_IMPORTS_D],
        },
        expected_by_kind={
            (1, "head", EdgeKind.IMPORTS): [],
            (2, "head", EdgeKind.IMPORTS): [_C_IMPORTS_D],
        },
    )
