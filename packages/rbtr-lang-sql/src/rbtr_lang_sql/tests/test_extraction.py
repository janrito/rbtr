"""SQL extraction tests.

Construct/mixed cases (`cases_extraction.py`) drive the shared checks;
the two functions at the end pin SQL-specific edge behaviour.
"""

from __future__ import annotations

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.domain.models import ChunkKind
from rbtr.testing import Indexed
from rbtr.testing.reads import outline


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(indexed: Indexed, files: dict[str, str], expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the outline."""
    with indexed(files) as build:
        reply = outline(build)
    symbols = [(s.kind, s.name, s.scope) for s in reply]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


@parametrize_with_cases(
    "files, expected_kinds, expected_methods", cases=".cases_extraction", has_tag="mixed"
)
def test_extracts_all_expected_kinds(
    indexed: Indexed,
    files: dict[str, str],
    expected_kinds: set[str],
    expected_methods: list[tuple[str, str]],
) -> None:
    """Realistic source produces all expected chunk kinds and method scoping."""
    with indexed(files) as build:
        reply = outline(build)
    kinds = {s.kind for s in reply}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(s.name, s.scope) for s in reply if s.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


def test_sql_pragma_is_a_parse_error_fragment(indexed: Indexed) -> None:
    """A DuckDB PRAGMA reaches a chunk only as a fragment of a parse error.

    The grammar has no PRAGMA statement node, so it parses to a top-level
    ERROR with no enclosing `statement` to capture. This is a known
    limitation guard; it flags the day the grammar gains PRAGMA support.
    """
    src = "PRAGMA create_fts_index('chunks', 'id', 'body');\n"
    with indexed({"input.sql": src}) as build:
        chunks = outline(build)
    assert [(c.kind, c.line_start, c.line_end) for c in chunks] == [(ChunkKind.FRAGMENT, 1, 1)]


@pytest.mark.xfail(
    strict=True,
    reason="tree-sitter-sql has no create_procedure node — parses to ERROR",
)
def test_stored_procedure_unsupported(indexed: Indexed) -> None:
    """A SQL stored procedure should ideally extract as a function (grammar gap).

    Strict xfail: when the grammar gains `create_procedure` this fails, prompting
    removal of the xfail and a sample update.
    """
    src = """\
CREATE PROCEDURE refresh()
LANGUAGE SQL
AS $$ DELETE FROM cache; $$;
"""
    with indexed({"input.sql": src}) as build:
        chunks = outline(build)
    assert (ChunkKind.FUNCTION, "refresh", "") in [(c.kind, c.name, c.scope) for c in chunks]


def test_sql_multi_statement_one_chunk_each(indexed: Indexed) -> None:
    """Each top-level statement in a file becomes its own chunk."""
    src = """\
CREATE TABLE a (id INT);
SELECT * FROM a;
DROP TABLE a;
"""
    with indexed({"input.sql": src}) as build:
        chunks = outline(build)
    assert [(c.kind, c.name) for c in chunks] == [
        (ChunkKind.CLASS, "a"),
        (ChunkKind.FUNCTION, "a"),
        (ChunkKind.FUNCTION, "a"),
    ]
