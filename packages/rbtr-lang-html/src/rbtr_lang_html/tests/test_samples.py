"""HTML sample extraction: the `samples/html/` project, built as rbtr builds a repository.

The sample links a `.js` and a `.css` and embeds inline `<script>`/`<style>`,
so the javascript and css plugins (dev dependencies) extract the linked and
injected chunks, and the edge snapshot captures the cross-file links.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rbtr.domain.models import ChunkKind

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from rbtr.daemon.dto import SymbolOut


@pytest.fixture(scope="module")
def sample_dir() -> Path:
    """The `samples/html/` project."""
    return Path(__file__).parent / "samples" / "html"


def test_emits_expected_kinds(sample_symbols: list[SymbolOut]) -> None:
    """The sample exercises HTML's import, doc-section, and (injected) function chunks."""
    assert {ChunkKind.IMPORT, ChunkKind.DOC_SECTION, ChunkKind.FUNCTION, ChunkKind.COMMENT} <= {
        s.kind for s in sample_symbols
    }


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
