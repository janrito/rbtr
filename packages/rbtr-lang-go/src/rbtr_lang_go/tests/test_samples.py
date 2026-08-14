"""go sample extraction: the `samples/go/` project through the real pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tree_sitter import Parser

from rbtr.domain.models import Chunk, ChunkKind, Edge
from rbtr.git import FileEntry
from rbtr.languages.edges import build_resolution_map, infer_import_edges
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager
from rbtr.testing import render_edges

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


@pytest.fixture
def project() -> list[tuple[str, str]]:
    root = Path(__file__).parent / "samples" / "go"
    return [
        (str(p.relative_to(root)), p.read_text()) for p in sorted(root.rglob("*")) if p.is_file()
    ]


@pytest.fixture
def sources(project: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The project's source files, which is every file but `go.mod`.

    A manifest is data the resolver reads, not source a grammar parses;
    a build chunks it as plaintext.
    """
    manager = get_manager()
    return [(path, text) for path, text in project if manager.detect_language(path)]


@pytest.fixture
def chunks(sources: list[tuple[str, str]]) -> list[Chunk]:
    manager = get_manager()
    out: list[Chunk] = []
    for path, text in sources:
        lang = manager.detect_language(path) or "go"
        out.extend(extract_file(FileEntry(path, "sha1", text.encode()), lang))
    return out


@pytest.fixture
def edges(project: list[tuple[str, str]], chunks: list[Chunk]) -> list[Edge]:
    manager = get_manager()
    repo_files = {path for path, _ in project}
    manifests = {path: text for path, text in project if path == "go.mod"}
    resolution = build_resolution_map(manager, manifests=manifests)
    return infer_import_edges(chunks, repo_files, resolution)


def test_emits_expected_kinds(chunks: list[Chunk]) -> None:
    kinds = {c.kind for c in chunks}
    assert {
        ChunkKind.FUNCTION,
        ChunkKind.METHOD,
        ChunkKind.CLASS,
        ChunkKind.VARIABLE,
        ChunkKind.IMPORT,
        ChunkKind.COMMENT,
    } <= kinds


def test_parses_cleanly(sources: list[tuple[str, str]]) -> None:
    manager = get_manager()
    for path, text in sources:
        grammar = manager.grammar(manager.detect_language(path) or "go")
        assert grammar is not None
        assert not Parser(grammar).parse(text.encode()).root_node.has_error, path


def test_extraction_matches_snapshot(chunks: list[Chunk], snapshot_json: SnapshotAssertion) -> None:
    assert chunks == snapshot_json


def test_edges_match_snapshot(
    chunks: list[Chunk], edges: list[Edge], snapshot_json: SnapshotAssertion
) -> None:
    assert render_edges(edges, chunks) == snapshot_json
