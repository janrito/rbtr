"""JS / TS / TSX sample extraction through the real pipeline.

Parametrised over the package's three ids. The `javascript` sample includes a
`styles.css`, extracted via the css plugin (a dev dependency), so the edge
snapshot captures the cross-file links.
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

_IDS = ["javascript", "typescript", "tsx"]


@pytest.fixture(params=_IDS)
def lang(request: pytest.FixtureRequest) -> str:
    """Each of the package's three language ids in turn."""
    return str(request.param)


@pytest.fixture
def project(lang: str) -> list[tuple[str, str]]:
    """The `(relative path, text)` files of the `samples/<lang>/` project."""
    root = Path(__file__).parent / "samples" / lang
    return [
        (str(p.relative_to(root)), p.read_text()) for p in sorted(root.rglob("*")) if p.is_file()
    ]


@pytest.fixture
def chunks(lang: str, project: list[tuple[str, str]]) -> list[Chunk]:
    """Chunks from every project file, each via the real `extract_file`.

    A file is extracted as *its own* detected language (the javascript sample
    spans js + css), so a project may mix languages.
    """
    manager = get_manager()
    out: list[Chunk] = []
    for path, text in project:
        file_lang = manager.detect_language(path) or lang
        out.extend(extract_file(FileEntry(path, "sha1", text.encode()), file_lang))
    return out


@pytest.fixture
def edges(project: list[tuple[str, str]], chunks: list[Chunk]) -> list[Edge]:
    """Import edges inferred across the project's files."""
    manager = get_manager()
    repo_files = {path for path, _ in project}
    return infer_import_edges(chunks, repo_files, build_resolution_map(manager))


def test_emits_expected_kinds(chunks: list[Chunk]) -> None:
    """Each sample exercises function, class, variable, and import chunks."""
    kinds = {c.kind for c in chunks}
    assert {
        ChunkKind.FUNCTION,
        ChunkKind.CLASS,
        ChunkKind.VARIABLE,
        ChunkKind.IMPORT,
        ChunkKind.COMMENT,
    } <= kinds


def test_parses_cleanly(lang: str, project: list[tuple[str, str]]) -> None:
    """Every project file is valid source — no tree-sitter ERROR/MISSING nodes."""
    manager = get_manager()
    for path, text in project:
        grammar = manager.load_grammar(manager.detect_language(path) or lang)
        assert grammar is not None
        assert not Parser(grammar).parse(text.encode()).root_node.has_error, path


def test_extraction_matches_snapshot(chunks: list[Chunk], snapshot_json: SnapshotAssertion) -> None:
    assert chunks == snapshot_json


def test_edges_match_snapshot(
    chunks: list[Chunk], edges: list[Edge], snapshot_json: SnapshotAssertion
) -> None:
    assert render_edges(edges, chunks) == snapshot_json
