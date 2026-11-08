"""SQL sample extraction: the `samples/sql/` project, built as rbtr builds a repository.

The snapshots are the golden record of what SQL extraction produces. The
`dialect` tests document how the single generic grammar handles each major
dialect. Engine-wide invariants are covered once in core.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest
from tree_sitter import Parser

from rbtr.daemon.dto import symbols
from rbtr.domain.models import ChunkKind
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager
from rbtr.languages.plaintext import PLAINTEXT
from rbtr.testing import render_edges, uncovered_lines

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from rbtr.domain.models import SnapshotRef
    from rbtr.index.index import Index


@pytest.fixture(scope="module")
def sample_dir() -> Path:
    """The `samples/sql/` project."""
    return Path(__file__).parent / "samples" / "sql"


def test_emits_expected_kinds(sample_index: Index, sample_ref: SnapshotRef) -> None:
    """The sample exercises SQL's class, function, and variable chunks."""
    kinds = {s.kind for s in symbols(sample_index.chunks(at=sample_ref))}
    assert {
        ChunkKind.CLASS,
        ChunkKind.FUNCTION,
        ChunkKind.VARIABLE,
        ChunkKind.COMMENT,
    } <= kinds


def test_parses_cleanly(sample_dir: Path, sample_index: Index, sample_ref: SnapshotRef) -> None:
    """Every source file is valid in the language the build read it as.

    A file no plugin claims is plaintext, which has no grammar to parse it.
    """
    manager = get_manager()
    files = (
        sample_index.chunks(at=sample_ref)
        .filter(pl.col("file_language") != PLAINTEXT)
        .select("file_path", "file_language")
        .unique()
    )
    for path, language in files.rows():
        grammar = manager.grammar(language)
        assert grammar is not None, path
        assert not Parser(grammar).parse((sample_dir / path).read_bytes()).root_node.has_error, path


def test_extraction_matches_snapshot(
    sample_index: Index, sample_ref: SnapshotRef, snapshot_json: SnapshotAssertion
) -> None:
    assert symbols(sample_index.chunks(at=sample_ref)) == snapshot_json


def test_edges_match_snapshot(
    sample_index: Index, sample_ref: SnapshotRef, snapshot_json: SnapshotAssertion
) -> None:
    chunks = sample_index.chunks(at=sample_ref)
    assert render_edges(sample_index.edges(within=[sample_ref]), chunks) == snapshot_json


def test_uncovered_lines_match_snapshot(
    sample_dir: Path, sample_index: Index, sample_ref: SnapshotRef, snapshot_json: SnapshotAssertion
) -> None:
    assert uncovered_lines(sample_dir, sample_index.chunks(at=sample_ref)) == snapshot_json


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
