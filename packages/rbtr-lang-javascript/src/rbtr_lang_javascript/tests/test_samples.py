"""JS / TS / TSX sample extraction: each sample project, built as rbtr builds a repository.

Parametrised over the package's three ids. The `javascript` sample includes a
`styles.css`, extracted via the css plugin (a dev dependency), so the edge
snapshot captures the cross-file links.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rbtr.domain.models import ChunkKind

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from rbtr.daemon.dto import SymbolOut


@pytest.fixture(scope="module", params=["javascript", "typescript", "tsx"])
def sample_dir(request: pytest.FixtureRequest) -> Path:
    """The `samples/<lang>/` project of each of the package's three language ids in turn."""
    return Path(__file__).parent / "samples" / str(request.param)


def test_emits_expected_kinds(sample_symbols: list[SymbolOut]) -> None:
    """Each sample exercises function, class, variable, and import chunks."""
    assert {
        ChunkKind.FUNCTION,
        ChunkKind.CLASS,
        ChunkKind.VARIABLE,
        ChunkKind.IMPORT,
        ChunkKind.COMMENT,
    } <= {s.kind for s in sample_symbols}


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
