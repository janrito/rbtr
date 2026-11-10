"""Tests for cross-file edge inference."""

from __future__ import annotations

import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from rbtr.languages.edges import EdgeChunkRow, infer_import_edges
from rbtr.languages.resolution import ImportResolution

from .cases_import_ways import ImportWayScenario


@pytest.fixture
def import_rules() -> dict[str, ImportResolution]:
    """How the neutral languages of `cases_import_ways` resolve: `code` imports, `prose` documents."""
    return {
        "code": ImportResolution(
            extensions=(".code",),
            index_files=("index.code",),
            source_roots=("",),
            path_substitutions=(),
        ),
        "prose": ImportResolution(
            extensions=(".code", ".prose"),
            index_files=(),
            source_roots=("",),
            path_substitutions=(),
            prose=True,
        ),
    }


@parametrize_with_cases("scenario", cases=".cases_import_ways")
def test_an_import_reaches_what_its_way_resolves_to(
    scenario: ImportWayScenario, import_rules: dict[str, ImportResolution]
) -> None:
    rows = pl.DataFrame(scenario.rows).pipe(EdgeChunkRow.validate, cast=True)
    edges = infer_import_edges(rows, set(scenario.repo_files), import_rules)
    assert set(edges.select("source_id", "target_id", pl.col("kind").cast(pl.String)).rows()) == (
        scenario.expected
    )


def test_a_name_twice_in_a_file_is_reached_at_its_first_line(
    import_rules: dict[str, ImportResolution],
) -> None:
    """Of two definitions an import names in one file, it reaches the one earlier in the file.

    The rows arrive later line first, so row order cannot decide it. A
    build cannot choose the order its rows arrive in, so these are the
    rows inference reads, written out.
    """
    rows = pl.DataFrame(
        {
            "id": ["import", "later", "earlier"],
            "file_path": ["a.code", "b.code", "b.code"],
            "line_start": [1, 5, 1],
            "kind": ["import", "function", "function"],
            "name": ["from b import f", "f", "f"],
            "language": ["code", "code", "code"],
            "module": ["b.code", None, None],
            "names": ["f", None, None],
            "dots": ["", None, None],
            "language_hint": ["", None, None],
        }
    ).pipe(EdgeChunkRow.validate, cast=True)

    edges = infer_import_edges(rows, {"a.code", "b.code"}, import_rules)

    assert edges["target_id"].to_list() == ["earlier"]
