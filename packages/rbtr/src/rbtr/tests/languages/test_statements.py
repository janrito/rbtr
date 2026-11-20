"""Declared statements, through the engine: a top-level statement a language declares is one chunk, and what is captured inside it is a chunk nested in it.

The language is Python's grammar under a query that declares a module's
`if` as a `@statement`, captures functions, comments and calls, and
nothing else, so what is not captured is left to fragments.
"""

from __future__ import annotations

import dataclasses

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.daemon.handlers import handle_find_refs
from rbtr.daemon.messages import FindRefsRequest
from rbtr.domain.models import ChunkKind
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration, QueryExtraction
from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_every_line_in_a_chunk
from rbtr.testing.reads import outline, repo_path
from rbtr_lang_python.plugin import python

from .cases_statements import StatementScenario


@pytest.fixture
def code() -> LanguageRegistration:
    """Python's grammar, as a language `code` of `.code` files: a module's `if` is a statement; functions, comments and calls are captured."""
    query = """\
(module (if_statement) @statement)
(function_definition name: (identifier) @_fn_name) @function
(comment) @comment
(call function: (identifier) @_call)
"""
    return dataclasses.replace(
        python, id="code", extensions=frozenset({".code"}), extraction=QueryExtraction(query=query)
    )


@parametrize_with_cases("scenario", cases=".cases_statements")
def test_a_files_symbols_around_its_top_level_statements(
    indexed: Indexed, code: LanguageRegistration, scenario: StatementScenario
) -> None:
    with get_manager().limited_to(code), indexed({"a.code": scenario.text}) as build:
        assert_every_line_in_a_chunk(build)
        reply = outline(build)
    assert [(s.kind.value, s.name, s.line_start, s.line_end) for s in reply] == scenario.symbols


def test_a_statement_captured_as_a_named_kind_too_is_that_kind(
    indexed: Indexed, code: LanguageRegistration
) -> None:
    """The `if` is also captured as a function named by its condition: one chunk, a function, and a narrowed overlap."""
    assert isinstance(code.extraction, QueryExtraction)
    query = code.extraction.query + "((if_statement condition: (identifier) @_fn_name) @function)\n"
    text = """\
if ready:
    go()
"""
    with (
        get_manager().limited_to(
            dataclasses.replace(code, extraction=QueryExtraction(query=query))
        ),
        indexed({"a.code": text}) as build,
    ):
        found = [(s.kind.value, s.name) for s in outline(build)]
        overlaps = [(o.kinds, o.kept, o.narrowed) for o in build.result.kind_overlaps]
    assert found == [("function", "ready")]
    assert overlaps == [((ChunkKind.FUNCTION, ChunkKind.STATEMENT), ChunkKind.FUNCTION, True)]


def test_a_call_in_a_declared_statement_is_the_statements(
    indexed: Indexed, code: LanguageRegistration
) -> None:
    """`main()` inside the `if` is the statement's call: `find-refs main` lists the statement."""
    text = """\
def main():
    pass


if __name__ == "__main__":
    main()
"""
    with get_manager().limited_to(code), indexed({"a.code": text}) as build:
        refs = handle_find_refs(
            FindRefsRequest(repo_path=repo_path(build), symbol="main"), build.index
        ).refs
    assert [(r.kind, r.line_start, r.edge.value) for r in refs] == [
        (ChunkKind.STATEMENT, 5, "calls")
    ]
