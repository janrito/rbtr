"""HCL sample extraction: the `samples/hcl/` project, built as rbtr builds a repository."""

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
    """The `samples/hcl/` project."""
    return Path(__file__).parent / "samples" / "hcl"


def test_emits_expected_kinds(sample_symbols: list[SymbolOut]) -> None:
    """The sample exercises HCL's config-key, comment and import chunks."""
    assert {ChunkKind.CONFIG_KEY, ChunkKind.COMMENT, ChunkKind.IMPORT} <= {
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
