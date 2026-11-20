"""HTML's script and stylesheet references, each a small project under `samples/imports/`, built as rbtr builds a repository.

- `scripts-and-a-stylesheet`: HTML resolves `<script src>` beside it and
  above it, and `<link href>` to a stylesheet.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


@pytest.fixture(
    scope="module",
    params=[
        "scripts-and-a-stylesheet",
    ],
)
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "imports" / str(request.param)


def test_edges_match_snapshot(sample_edges: list[str], snapshot_json: SnapshotAssertion) -> None:
    assert sample_edges == snapshot_json
