"""Markdown sample extraction: the `samples/markdown/` project, built as rbtr builds a repository.

The sample embeds fenced code (python, sh, yaml, toml, html, svelte), delegated to
those plugins, and links between files, so the snapshot captures the doc sections,
injected chunks, and cross-file edges.
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
    """The `samples/markdown/` project."""
    return Path(__file__).parent / "samples" / "markdown"


def test_emits_expected_kinds(sample_symbols: list[SymbolOut]) -> None:
    """The sample exercises Markdown's doc-section, import, and (injected) function chunks."""
    assert {ChunkKind.DOC_SECTION, ChunkKind.IMPORT, ChunkKind.FUNCTION} <= {
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
