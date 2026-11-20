"""Tests for cross-file edge inference."""

from __future__ import annotations

import polars as pl
import pygit2
import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.domain.models import SnapshotRef
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.languages.edges import EdgeChunkRow, infer_import_edges
from rbtr.languages.manager import get_manager
from rbtr.languages.resolution import build_resolution_map
from rbtr.testing import render_edges

from ..conftest import make_commit
from .cases_edges import EdgeScenario


@fixture
@parametrize_with_cases("scenario", cases=".cases_edges")
def edge_scenario(scenario: EdgeScenario) -> EdgeScenario:
    return scenario


@pytest.fixture
def edge_scenario_ref(
    edge_scenario: EdgeScenario, index: Index, git_repo: pygit2.Repository, repo_path: str
) -> SnapshotRef:
    sha = str(make_commit(git_repo, edge_scenario.files, ref="refs/heads/case"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


def test_a_build_infers_the_edges_its_imports_make(
    edge_scenario: EdgeScenario, edge_scenario_ref: SnapshotRef, index: Index
) -> None:
    chunks = index.chunks(at=edge_scenario_ref)
    edges = index.edges(within=[edge_scenario_ref])
    assert render_edges(edges, chunks) == edge_scenario.expected


def test_a_name_twice_in_a_file_is_reached_at_its_first_line() -> None:
    """Of two definitions an import names in one file, it reaches the one earlier in the file.

    The rows arrive later line first, so row order cannot decide it. A
    build cannot choose the order its rows arrive in, so these are the
    rows inference reads, written out.
    """
    rows = pl.DataFrame(
        {
            "id": ["import", "later", "earlier"],
            "file_path": ["a.py", "b.py", "b.py"],
            "line_start": [1, 5, 1],
            "kind": ["import", "function", "function"],
            "name": ["from b import f", "f", "f"],
            "language": ["python", "python", "python"],
            "module": ["b", None, None],
            "names": ["f", None, None],
            "dots": ["", None, None],
            "language_hint": ["", None, None],
        }
    ).pipe(EdgeChunkRow.validate, cast=True)

    edges = infer_import_edges(rows, {"a.py", "b.py"}, build_resolution_map(get_manager()))

    assert edges["target_id"].to_list() == ["earlier"]
