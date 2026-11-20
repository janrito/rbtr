"""Tests for cross-file edge inference."""

from __future__ import annotations

import dataframely as dy
import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from rbtr.domain.edge_frames import (
    EdgeChunkRow,
    LanguageResolutionRow,
    MemberOrder,
    ModuleStyle,
    SectionNames,
)
from rbtr.domain.edges import infer_edges
from rbtr.languages.registration import LanguageRegistration
from rbtr.languages.resolution import language_resolutions

from .cases_calls import CallScenario
from .cases_import_ways import ImportWayScenario


@pytest.fixture
def resolutions() -> dy.DataFrame[LanguageResolutionRow]:
    """How the neutral languages of `cases_import_ways` and `cases_calls` resolve.

    `code` imports what it names; `prose` and `page` document it. A
    reference to a `prose` section compares folded, and one to a `page`
    section as written. `dotted` writes modules dotted, as `code` writes
    paths. In `members` a method's unqualified names reach its class's
    members; in `package` a directory's files share their names; in
    `linked` a function is one across files by its scope and name. `code`
    searches a class's bases first listed first, `reversed` last listed
    first, and `nearest` the shallowest first.
    """
    return language_resolutions(
        [
            LanguageRegistration(
                id="code", extensions=frozenset({".code"}), index_files=frozenset({"index.code"})
            ),
            LanguageRegistration(
                id="dotted",
                extensions=frozenset({".dot"}),
                index_files=frozenset({"init.dot"}),
                module_style=ModuleStyle.DOTTED,
            ),
            LanguageRegistration(
                id="members", extensions=frozenset({".mem"}), members_in_scope=True
            ),
            LanguageRegistration(
                id="package", extensions=frozenset({".pkg"}), directory_scope=True
            ),
            LanguageRegistration(
                id="linked", extensions=frozenset({".lnk"}), linked_functions=True
            ),
            LanguageRegistration(
                id="reversed", extensions=frozenset({".rev"}), member_order=MemberOrder.REVERSED
            ),
            LanguageRegistration(
                id="nearest", extensions=frozenset({".near"}), member_order=MemberOrder.NEAREST
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
    edges = infer_edges(rows, set(scenario.repo_files), resolutions, steps=8)
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
            "scope": [""] * 3,
            "bindings": [None] * 3,
            "refs": [None] * 3,
        }
    ).pipe(EdgeChunkRow.validate, cast=True)

    edges = infer_edges(rows, {"a.code", "b.code"}, resolutions, steps=8)

    assert edges["target_id"].to_list() == ["earlier"]


@parametrize_with_cases("scenario", cases=".cases_calls")
def test_a_call_or_base_reaches_the_definition_it_names(
    scenario: CallScenario, resolutions: dy.DataFrame[LanguageResolutionRow]
) -> None:
    rows = pl.DataFrame(scenario.rows, schema=EdgeChunkRow.create_empty().schema).pipe(
        EdgeChunkRow.validate, cast=True
    )
    edges = infer_edges(rows, set(scenario.repo_files), resolutions, steps=8).filter(
        pl.col("kind").is_in(["calls", "inherits"])
    )
    assert set(edges.select("source_id", "target_id", pl.col("kind").cast(pl.String)).rows()) == (
        scenario.expected
    )
    if scenario.located is not None:
        assert set(edges.select("source_path", "target_path").rows()) == scenario.located
