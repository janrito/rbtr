"""Scenarios for edge storage and retrieval.

Cases take named edge fixtures from ``conftest.py``.  No
module-level constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Edge, EdgeKind


@dataclass(frozen=True)
class RepoEdges:
    """Edges scoped to a single (repo, commit) pair."""

    per_commit: dict[str, list[Edge]] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeScenario:
    """Declarative edge-family test data."""

    repo_paths: list[str] = field(default_factory=lambda: ["/default"])
    per_repo: list[RepoEdges] = field(default_factory=list)
    expected_edges: dict[tuple[int, str], list[Edge]] = field(default_factory=dict)
    expected_by_source: dict[tuple[int, str, str], list[Edge]] = field(default_factory=dict)
    expected_by_kind: dict[tuple[int, str, EdgeKind], list[Edge]] = field(default_factory=dict)


def case_empty_store() -> EdgeScenario:
    return EdgeScenario(repo_paths=[])


def case_single_edge_single_commit(edge_math_calls_class: Edge) -> EdgeScenario:
    return EdgeScenario(
        repo_paths=["/r"],
        per_repo=[RepoEdges(per_commit={"head": [edge_math_calls_class]})],
        expected_edges={
            (1, "head"): [edge_math_calls_class],
            (1, "other"): [],
        },
    )


def case_two_edges_mixed_kinds(edge_a_calls_b: Edge, edge_c_imports_d: Edge) -> EdgeScenario:
    edges = [edge_a_calls_b, edge_c_imports_d]
    return EdgeScenario(
        repo_paths=["/r"],
        per_repo=[RepoEdges(per_commit={"head": list(edges)})],
        expected_edges={(1, "head"): list(edges)},
        expected_by_source={
            (1, "head", "a"): [edge_a_calls_b],
            (1, "head", "c"): [edge_c_imports_d],
            (1, "head", "nonexistent"): [],
        },
        expected_by_kind={
            (1, "head", EdgeKind.CALLS): [edge_a_calls_b],
            (1, "head", EdgeKind.IMPORTS): [edge_c_imports_d],
        },
    )


def case_same_edge_on_two_commits(edge_a_calls_b: Edge) -> EdgeScenario:
    return EdgeScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoEdges(
                per_commit={
                    "a": [edge_a_calls_b],
                    "b": [edge_a_calls_b],
                }
            )
        ],
        expected_edges={
            (1, "a"): [edge_a_calls_b],
            (1, "b"): [edge_a_calls_b],
        },
    )


def case_two_repos_isolated(edge_a_calls_b: Edge, edge_c_imports_d: Edge) -> EdgeScenario:
    return EdgeScenario(
        repo_paths=["/r1", "/r2"],
        per_repo=[
            RepoEdges(per_commit={"head": [edge_a_calls_b]}),
            RepoEdges(per_commit={"head": [edge_c_imports_d]}),
        ],
        expected_edges={
            (1, "head"): [edge_a_calls_b],
            (2, "head"): [edge_c_imports_d],
        },
        expected_by_kind={
            (1, "head", EdgeKind.IMPORTS): [],
            (2, "head", EdgeKind.IMPORTS): [edge_c_imports_d],
        },
    )
