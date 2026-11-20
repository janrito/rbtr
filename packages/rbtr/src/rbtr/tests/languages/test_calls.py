"""Calls and bases, through the engine: files built as rbtr builds a repository, read back with `find-refs`."""

from __future__ import annotations

import dataclasses

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.domain.models import EdgeKind
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration
from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_reached
from rbtr_lang_python.plugin import python

from .cases_calls import CallScenario


@pytest.fixture
def code_languages() -> list[LanguageRegistration]:
    """`code`, Python's grammar and query under `.code`, and `pkg`, the same with its directories' files sharing their names."""
    code = dataclasses.replace(
        python, id="code", extensions=frozenset({".code"}), index_files=frozenset({"__init__.code"})
    )
    return [
        code,
        dataclasses.replace(code, id="pkg", extensions=frozenset({".pkg"}), directory_scope=True),
    ]


@parametrize_with_cases("scenario", cases=".cases_calls")
def test_a_call_or_base_reaches_the_definition_it_names(
    indexed: Indexed,
    code_languages: list[LanguageRegistration],
    scenario: CallScenario,
) -> None:
    with get_manager().limited_to(*code_languages), indexed(scenario.files) as build:
        assert_reached(build, scenario.reached, edges={EdgeKind.CALLS, EdgeKind.INHERITS})
