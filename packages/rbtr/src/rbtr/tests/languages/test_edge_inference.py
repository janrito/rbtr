"""Imports, links and references, through the engine; and the one rule of row order no build can set up."""

from __future__ import annotations

import dataclasses

import dataframely as dy
import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from rbtr.domain.edge_frames import EdgeChunkRow, LanguageResolutionRow
from rbtr.domain.edges import infer_edges
from rbtr.domain.models import SectionNames
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration
from rbtr.languages.resolution import language_resolutions
from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_reached
from rbtr_lang_javascript.plugin import javascript
from rbtr_lang_markdown.plugin import markdown, markdown_inline
from rbtr_lang_python.plugin import python
from rbtr_lang_rst.plugin import rst

from .cases_import_ways import ImportWayScenario


@pytest.fixture
def prose_languages() -> list[LanguageRegistration]:
    """The installed Python, Markdown, rST and JavaScript, and `page`: Markdown under `.page`, its sections compared as written."""
    page = dataclasses.replace(
        markdown, id="page", extensions=frozenset({".page"}), section_names=SectionNames.EXACT
    )
    return [python, markdown, markdown_inline, rst, javascript, page]


@parametrize_with_cases("scenario", cases=".cases_import_ways")
def test_an_import_reaches_what_its_way_resolves_to(
    indexed: Indexed,
    prose_languages: list[LanguageRegistration],
    scenario: ImportWayScenario,
) -> None:
    with get_manager().limited_to(*prose_languages), indexed(scenario.files) as build:
        assert_reached(build, scenario.reached)


@pytest.fixture
def resolutions() -> dy.DataFrame[LanguageResolutionRow]:
    """`code`, a language of `.code` files, imports what it names."""
    return language_resolutions([LanguageRegistration(id="code", extensions=frozenset({".code"}))])


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
            "scope": [""] * 3,
            "imports": [
                [
                    {
                        "text": "from b import f",
                        "module": "b.code",
                        "names": "f",
                        "language_hint": "",
                        "bindings": [],
                    }
                ],
                [],
                [],
            ],
            "refs": [None] * 3,
        }
    ).pipe(EdgeChunkRow.validate, cast=True)

    edges = infer_edges(rows, {"a.code", "b.code"}, resolutions, steps=8)

    assert edges["target_id"].to_list() == ["earlier"]
