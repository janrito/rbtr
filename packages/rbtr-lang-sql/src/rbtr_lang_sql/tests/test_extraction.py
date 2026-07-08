"""SQL extraction tests.

Construct/mixed cases (`cases_extraction.py`) drive the shared checks;
the two functions at the end pin SQL-specific edge behaviour.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.index.models import ChunkKind
from rbtr.languages.extract import extract_file


@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


@parametrize_with_cases(
    "lang, source, expected_kinds, expected_methods", cases=".cases_extraction", has_tag="mixed"
)
def test_extracts_all_expected_kinds(
    lang: str,
    source: str,
    expected_kinds: set[str],
    expected_methods: list[tuple[str, str]],
) -> None:
    """Realistic source produces all expected chunk kinds and method scoping."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    kinds = {c.kind for c in chunks}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(c.name, c.scope) for c in chunks if c.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


def test_sql_pragma_not_extracted() -> None:
    """A DuckDB PRAGMA yields no definition chunk.

    The grammar has no PRAGMA statement node, so it parses to a top-level
    ERROR with no enclosing `statement` to capture. This is a known
    limitation guard; it flags the day the grammar gains PRAGMA support.
    Only the content-less host-presence chunk (for blob dedup) remains.
    """
    src = "PRAGMA create_fts_index('chunks', 'id', 'body');\n"
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "sql")
    assert [c for c in chunks if c.content] == []


def test_sql_multi_statement_one_chunk_each() -> None:
    """Each top-level statement in a file becomes its own chunk."""
    src = """\
CREATE TABLE a (id INT);
SELECT * FROM a;
DROP TABLE a;
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "sql")
    assert [(c.kind, c.name) for c in chunks] == [
        (ChunkKind.CLASS, "a"),
        (ChunkKind.FUNCTION, "a"),
        (ChunkKind.FUNCTION, "a"),
    ]
