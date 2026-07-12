"""C++ sample extraction: the `samples/cpp/` project through the real pipeline."""

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
    root = Path(__file__).parent / "samples" / "cpp"
    return [
        (str(p.relative_to(root)), p.read_text()) for p in sorted(root.rglob("*")) if p.is_file()
    ]


@pytest.fixture
def chunks(project: list[tuple[str, str]]) -> list[Chunk]:
    manager = get_manager()
    out: list[Chunk] = []
    for path, text in project:
        lang = manager.detect_language(path) or "cpp"
        out.extend(extract_file(FileEntry(path, "sha1", text.encode()), lang))
    return out


@pytest.fixture
def edges(project: list[tuple[str, str]], chunks: list[Chunk]) -> list[Edge]:
    manager = get_manager()
    repo_files = {path for path, _ in project}
    return infer_import_edges(chunks, repo_files, build_resolution_map(manager))


def test_emits_expected_kinds(chunks: list[Chunk]) -> None:
    kinds = {c.kind for c in chunks}
    assert {
        ChunkKind.FUNCTION,
        ChunkKind.CLASS,
        ChunkKind.METHOD,
        ChunkKind.VARIABLE,
        ChunkKind.IMPORT,
    } <= kinds


def test_parses_cleanly(project: list[tuple[str, str]]) -> None:
    manager = get_manager()
    for path, text in project:
        grammar = manager.load_grammar(manager.detect_language(path) or "cpp")
        assert grammar is not None
        assert not Parser(grammar).parse(text.encode()).root_node.has_error, path


def test_extraction_matches_snapshot(chunks: list[Chunk], snapshot_json: SnapshotAssertion) -> None:
    assert chunks == snapshot_json


def test_edges_match_snapshot(
    chunks: list[Chunk], edges: list[Edge], snapshot_json: SnapshotAssertion
) -> None:
    assert render_edges(edges, chunks) == snapshot_json
