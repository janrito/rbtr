"""Levelled headings, through the engine: a file built as rbtr builds a repository, read back with `list-symbols`."""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.languages.manager import get_manager
from rbtr.testing import Indexed
from rbtr.testing.reads import outline
from rbtr_lang_markdown.plugin import markdown, markdown_inline
from rbtr_lang_rst.plugin import rst

from .cases_levelled import LevelledScenario


@parametrize_with_cases("scenario", cases=".cases_levelled")
def test_a_heading_ends_at_the_next_and_sits_under_a_higher_one(
    indexed: Indexed, scenario: LevelledScenario
) -> None:
    with (
        get_manager().limited_to(markdown, markdown_inline, rst),
        indexed({scenario.path: scenario.text}) as index,
    ):
        reply = outline(index, [scenario.path])
    assert [
        (s.kind.value, s.name, s.scope, s.line_start, s.line_end) for s in reply
    ] == scenario.symbols
