"""SQL sample extraction: the `samples/sql/` project through the real pipeline.

The snapshots are the golden record of what SQL extraction produces. The
`dialect` tests document how the single generic grammar handles each major
dialect. Engine-wide invariants are covered once in core.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tree_sitter import Parser

from rbtr.git import FileEntry
from rbtr.index.models import Chunk, ChunkKind, Edge
from rbtr.languages.edges import build_resolution_map, infer_import_edges
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager
from rbtr.testing import render_edges

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


@pytest.fixture
def project() -> list[tuple[str, str]]:
    """The `(relative path, text)` files of the `samples/sql/` project."""
    root = Path(__file__).parent / "samples" / "sql"
    return [
        (str(p.relative_to(root)), p.read_text()) for p in sorted(root.rglob("*")) if p.is_file()
    ]


@pytest.fixture
def chunks(project: list[tuple[str, str]]) -> list[Chunk]:
    """Chunks from every project file, each via the real `extract_file`."""
    manager = get_manager()
    out: list[Chunk] = []
    for path, text in project:
        lang = manager.detect_language(path) or "sql"
        out.extend(extract_file(FileEntry(path, "sha1", text.encode()), lang))
    return out


@pytest.fixture
def edges(project: list[tuple[str, str]], chunks: list[Chunk]) -> list[Edge]:
    """Import edges inferred across the project's files."""
    manager = get_manager()
    repo_files = {path for path, _ in project}
    return infer_import_edges(chunks, repo_files, build_resolution_map(manager))


def test_emits_expected_kinds(chunks: list[Chunk]) -> None:
    """The sample exercises SQL's class, function, and variable chunks."""
    kinds = {c.kind for c in chunks}
    assert {
        ChunkKind.CLASS,
        ChunkKind.FUNCTION,
        ChunkKind.VARIABLE,
        ChunkKind.COMMENT,
    } <= kinds


def test_parses_cleanly(project: list[tuple[str, str]]) -> None:
    """Every project file is valid source — no tree-sitter ERROR/MISSING nodes."""
    manager = get_manager()
    for path, text in project:
        grammar = manager.load_grammar(manager.detect_language(path) or "sql")
        assert grammar is not None
        assert not Parser(grammar).parse(text.encode()).root_node.has_error, path


def test_extraction_matches_snapshot(chunks: list[Chunk], snapshot_json: SnapshotAssertion) -> None:
    assert chunks == snapshot_json


def test_edges_match_snapshot(
    chunks: list[Chunk], edges: list[Edge], snapshot_json: SnapshotAssertion
) -> None:
    assert render_edges(edges, chunks) == snapshot_json


# ── SQL dialects ─────────────────────────────────────────────────────
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
    grammar = get_manager().load_grammar("sql")
    assert grammar is not None
    assert not Parser(grammar).parse(source.encode()).root_node.has_error
