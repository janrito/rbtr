"""HTML sample extraction: the `samples/html/` project, built as rbtr builds a repository.

The sample links a `.js` and a `.css` and embeds inline `<script>`/`<style>`,
so the javascript and css plugins (dev dependencies) extract the linked and
injected chunks, and the edge snapshot captures the cross-file links.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest
from tree_sitter import Parser

from rbtr.daemon.dto import symbols
from rbtr.domain.models import ChunkKind
from rbtr.languages.manager import get_manager
from rbtr.languages.plaintext import PLAINTEXT
from rbtr.testing import render_edges, uncovered_lines

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from rbtr.domain.models import SnapshotRef
    from rbtr.index.index import Index


@pytest.fixture(scope="module")
def sample_dir() -> Path:
    """The `samples/html/` project."""
    return Path(__file__).parent / "samples" / "html"


def test_emits_expected_kinds(sample_index: Index, sample_ref: SnapshotRef) -> None:
    """The sample exercises HTML's import, doc-section, and (injected) function chunks."""
    kinds = {s.kind for s in symbols(sample_index.chunks(at=sample_ref))}
    assert {ChunkKind.IMPORT, ChunkKind.DOC_SECTION, ChunkKind.FUNCTION, ChunkKind.COMMENT} <= kinds


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
