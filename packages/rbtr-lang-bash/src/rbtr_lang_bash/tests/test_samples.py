"""Bash sample extraction: the `samples/bash/` project through the real pipeline.

The snapshots are the golden record of what bash extraction produces;
regenerate with `pytest --snapshot-update` after an intended change.
Engine-wide invariants (determinism, line numbers, syntax-error recovery)
are covered once in core, not re-run per language.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tree_sitter import Parser

from rbtr.git import FileEntry
from rbtr.index.edges import build_resolution_map, infer_import_edges
from rbtr.index.models import Chunk, ChunkKind, Edge
from rbtr.index.orchestrator import extract_file
from rbtr.languages.manager import get_manager
from rbtr.testing import render_edges

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


@pytest.fixture
def project() -> list[tuple[str, str]]:
    """The `(relative path, text)` files of the `samples/bash/` project."""
    root = Path(__file__).parent / "samples" / "bash"
    return [
        (str(p.relative_to(root)), p.read_text()) for p in sorted(root.rglob("*")) if p.is_file()
    ]


@pytest.fixture
def chunks(project: list[tuple[str, str]]) -> list[Chunk]:
    """Chunks from every project file, each via the real `extract_file`."""
    manager = get_manager()
    out: list[Chunk] = []
    for path, text in project:
        lang = manager.detect_language(path) or "bash"
        out.extend(extract_file(FileEntry(path, "sha1", text.encode()), lang))
    return out


@pytest.fixture
def edges(project: list[tuple[str, str]], chunks: list[Chunk]) -> list[Edge]:
    """Import edges inferred across the project's files."""
    manager = get_manager()
    return infer_import_edges(chunks, {p for p, _ in project}, build_resolution_map(manager))


def test_emits_expected_kinds(chunks: list[Chunk]) -> None:
    """The sample exercises bash's function, variable, and import chunks."""
    kinds = {c.kind for c in chunks}
    assert {ChunkKind.FUNCTION, ChunkKind.VARIABLE, ChunkKind.IMPORT} <= kinds


def test_parses_cleanly(project: list[tuple[str, str]]) -> None:
    """Every project file is valid source — no tree-sitter ERROR/MISSING nodes."""
    manager = get_manager()
    for path, text in project:
        grammar = manager.load_grammar(manager.detect_language(path) or "bash")
        assert grammar is not None
        assert not Parser(grammar).parse(text.encode()).root_node.has_error, path


def test_extraction_matches_snapshot(chunks: list[Chunk], snapshot_json: SnapshotAssertion) -> None:
    assert chunks == snapshot_json


def test_edges_match_snapshot(
    chunks: list[Chunk], edges: list[Edge], snapshot_json: SnapshotAssertion
) -> None:
    assert render_edges(edges, chunks) == snapshot_json
