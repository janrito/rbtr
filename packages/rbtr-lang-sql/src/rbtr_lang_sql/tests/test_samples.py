"""SQL sample extraction: the `samples/sql/` project, built as rbtr builds a repository.

The snapshots are the golden record of what SQL extraction produces. The
`dialect` tests document how the single generic grammar handles each major
dialect. Engine-wide invariants are covered once in core.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tree_sitter import Parser

from rbtr.domain.models import ChunkKind
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from rbtr.daemon.dto import SymbolOut


@pytest.fixture(scope="module")
def sample_dir() -> Path:
    """The `samples/sql/` project."""
    return Path(__file__).parent / "samples" / "sql"


def test_emits_expected_kinds(sample_symbols: list[SymbolOut]) -> None:
    """The sample exercises SQL's class, function, and variable chunks."""
    assert {
        ChunkKind.CLASS,
        ChunkKind.FUNCTION,
        ChunkKind.VARIABLE,
        ChunkKind.COMMENT,
    } <= {s.kind for s in sample_symbols}


def test_parses_cleanly(sample_parse_errors: list[str]) -> None:
    """Every source file is valid in the language the build read it as."""
    assert sample_parse_errors == []


def test_extraction_matches_snapshot(
    sample_symbols: list[SymbolOut], snapshot_json: SnapshotAssertion
) -> None:
    assert sample_symbols == snapshot_json


def test_edges_match_snapshot(sample_edges: list[str], snapshot_json: SnapshotAssertion) -> None:
    assert sample_edges == snapshot_json


def test_uncovered_lines_match_snapshot(
    sample_uncovered_lines: list[str], snapshot_json: SnapshotAssertion
) -> None:
    assert sample_uncovered_lines == snapshot_json


# rbtr ships one generic SQL grammar for all `.sql` files. These pin
# current extraction per dialect; the parse-clean test is a strict-xfail
# sentinel that fires when the grammar gains full support for a dialect.


@pytest.mark.parametrize(
    "dialect", ["sql_postgres", "sql_mysql", "sql_sqlite", "sql_duckdb", "sql_clickhouse"]
)
def test_sql_dialect_extraction_matches_snapshot(
    dialect: str, snapshot_json: SnapshotAssertion
) -> None:
    """Current extraction for each SQL dialect under the generic grammar."""
    source = (Path(__file__).parent / "samples" / f"{dialect}.sql").read_text()
    chunks = list(extract_file(FileEntry(f"{dialect}.sql", "sha1", source.encode()), "sql"))
    assert chunks == snapshot_json


@pytest.mark.xfail(
    reason="generic SQL grammar does not fully parse dialect-specific syntax",
    strict=True,
)
@pytest.mark.parametrize(
    "dialect", ["sql_postgres", "sql_mysql", "sql_sqlite", "sql_duckdb", "sql_clickhouse"]
)
def test_sql_dialect_parses_cleanly(dialect: str) -> None:
    """Sentinel: flips to XPASS (failing) when a dialect parses cleanly."""
    source = (Path(__file__).parent / "samples" / f"{dialect}.sql").read_text()
    grammar = get_manager().grammar("sql")
    assert grammar is not None
    assert not Parser(grammar).parse(source.encode()).root_node.has_error
