"""HTML sample extraction: the `samples/html/` project through the real pipeline.

The sample links a `.js` and a `.css` and embeds inline `<script>`/`<style>`,
so the javascript and css plugins (dev dependencies) extract the linked and
injected chunks, and the edge snapshot captures the cross-file links.
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
    """The `(relative path, text)` files of the `samples/html/` project."""
    root = Path(__file__).parent / "samples" / "html"
    return [
        (str(p.relative_to(root)), p.read_text()) for p in sorted(root.rglob("*")) if p.is_file()
    ]


@pytest.fixture
def chunks(project: list[tuple[str, str]]) -> list[Chunk]:
    """Chunks from every project file, each via the real `extract_file`.

    A file is extracted as its own detected language, so the project spans
    html, javascript, and css.
    """
    manager = get_manager()
    out: list[Chunk] = []
    for path, text in project:
        lang = manager.detect_language(path) or "html"
        out.extend(extract_file(FileEntry(path, "sha1", text.encode()), lang))
    return out


@pytest.fixture
def edges(project: list[tuple[str, str]], chunks: list[Chunk]) -> list[Edge]:
    """Import edges inferred across the project's files."""
    manager = get_manager()
    repo_files = {path for path, _ in project}
    return infer_import_edges(chunks, repo_files, build_resolution_map(manager))


def test_emits_expected_kinds(chunks: list[Chunk]) -> None:
    """The sample exercises HTML's import, doc-section, and (injected) function chunks."""
    kinds = {c.kind for c in chunks}
    assert {ChunkKind.IMPORT, ChunkKind.DOC_SECTION, ChunkKind.FUNCTION} <= kinds


def test_parses_cleanly(project: list[tuple[str, str]]) -> None:
    """Every project file is valid source — no tree-sitter ERROR/MISSING nodes."""
    manager = get_manager()
    for path, text in project:
        grammar = manager.load_grammar(manager.detect_language(path) or "html")
        assert grammar is not None
        assert not Parser(grammar).parse(text.encode()).root_node.has_error, path


def test_extraction_matches_snapshot(chunks: list[Chunk], snapshot_json: SnapshotAssertion) -> None:
    assert chunks == snapshot_json


def test_edges_match_snapshot(
    chunks: list[Chunk], edges: list[Edge], snapshot_json: SnapshotAssertion
) -> None:
    assert render_edges(edges, chunks) == snapshot_json
