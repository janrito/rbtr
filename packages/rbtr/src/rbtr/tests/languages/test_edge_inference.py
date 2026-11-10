"""Tests for cross-file edge inference."""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.domain.models import (
    Chunk,
    Edge,
    EdgeKind,
)
from rbtr.git import FileEntry
from rbtr.languages.edges import (
    infer_import_edges,
)
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager
from rbtr.languages.resolution import build_resolution_map
from rbtr.testing import edge_rows

from .cases_edges import EdgeScenario


@fixture
@parametrize_with_cases("scenario", cases=".cases_edges")
def inferred(scenario: EdgeScenario) -> tuple[EdgeScenario, list[Edge], dict[str, str]]:
    """Build the scenario's chunks and keep each spec handle's real id.

    A `ChunkSpec.id` is a handle a case can refer to; a chunk's actual id
    is derived from its content, so expectations written in handles are
    translated through this map.
    """
    chunks = [
        Chunk.model_validate(
            {
                "blob_sha": f"sha_{spec.id}",
                "file_path": spec.file_path,
                "role": {"kind": spec.kind, "name": spec.name, "meta": spec.metadata},
                "language": spec.language,
                "content": spec.content,
                "span": {"start": 1, "end": 1},
            }
        )
        for spec in scenario.chunks
    ]
    ids = {spec.id: chunk.id for spec, chunk in zip(scenario.chunks, chunks, strict=True)}
    edges = infer_import_edges(edge_rows(chunks), set(scenario.repo_files), scenario.resolution_map)
    return scenario, edges, ids


def test_inference_produces_expected_edges(
    inferred: tuple[EdgeScenario, list[Edge], dict[str, str]],
) -> None:
    scenario, edges, ids = inferred
    pairs = frozenset((e.source_id, e.target_id) for e in edges)
    expected = frozenset((ids[src], ids[dst]) for src, dst in scenario.expected)
    assert pairs == expected


def test_inference_edge_kind_is_correct(
    inferred: tuple[EdgeScenario, list[Edge], dict[str, str]],
) -> None:
    scenario, edges, _ids = inferred
    if not edges:
        return
    expected_kind = scenario.expected_edge_kind or EdgeKind.IMPORTS
    for e in edges:
        assert e.kind == expected_kind


def test_a_name_twice_in_a_file_is_reached_at_its_first_line() -> None:
    """Of two definitions an import names in one file, it reaches the one earlier in the file.

    The rows arrive later line first, so row order cannot decide it.
    """
    importer = extract_file(FileEntry("a.py", "sha_a", b"from b import f\n"), "python")
    target_source = b"""\
def f():
    return 1


def f():
    return 2
"""
    target = extract_file(FileEntry("b.py", "sha_b", target_source), "python")
    rows = edge_rows([*importer, *target][::-1])

    edges = infer_import_edges(rows, {"a.py", "b.py"}, build_resolution_map(get_manager()))

    assert [e.target_id for e in edges] == [c.id for c in target if c.span.start == 1]
