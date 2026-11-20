"""Tests for cross-file edge inference."""

from __future__ import annotations

import dataframely as dy
import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from rbtr.domain.edge_frames import (
    EdgeChunkRow,
    LanguageResolutionRow,
    SectionNames,
)
from rbtr.domain.edges import infer_edges
from rbtr.languages.registration import LanguageRegistration
from rbtr.languages.resolution import language_resolutions

from .cases_import_ways import ImportWayScenario


@pytest.fixture
def resolutions() -> dy.DataFrame[LanguageResolutionRow]:
    """How the neutral languages of `cases_import_ways` resolve.

    `code` imports what it names; `prose` and `page` document it. A
    reference to a `prose` section compares folded, and one to a `page`
    section as written.
    """
    return language_resolutions(
        [
            LanguageRegistration(
                id="code", extensions=frozenset({".code"}), index_files=frozenset({"index.code"})
            ),
            LanguageRegistration(
                id="prose",
                extensions=frozenset({".prose"}),
                import_targets=frozenset({"prose", "code"}),
                prose=True,
                section_names=SectionNames.FOLDED,
            ),
            LanguageRegistration(id="page", extensions=frozenset({".page"}), prose=True),
        ]
    )


@parametrize_with_cases("scenario", cases=".cases_import_ways")
def test_an_import_reaches_what_its_way_resolves_to(
    scenario: ImportWayScenario, resolutions: dy.DataFrame[LanguageResolutionRow]
) -> None:
    rows = pl.DataFrame(scenario.rows, schema=EdgeChunkRow.create_empty().schema).pipe(
        EdgeChunkRow.validate, cast=True
    )
    edges = infer_edges(rows, set(scenario.repo_files), resolutions)
    assert set(edges.select("source_id", "target_id", pl.col("kind").cast(pl.String)).rows()) == (
        scenario.expected
    )


def test_a_name_twice_in_a_file_is_reached_at_its_first_line(
    resolutions: dy.DataFrame[LanguageResolutionRow],
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
            "language_hint": ["", None, None],
        }
    ).pipe(EdgeChunkRow.validate, cast=True)

    edges = infer_edges(rows, {"a.code", "b.code"}, resolutions)

    assert edges["target_id"].to_list() == ["earlier"]
