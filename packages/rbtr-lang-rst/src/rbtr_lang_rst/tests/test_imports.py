"""rST's links, each a small project under `samples/imports/`, built as rbtr builds a repository.

- `title-ids`: A hyperlink written with the id docutils gives a section's
  title documents that section. The ids are what docutils 0.23 itself
  gave the guide's titles.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


@pytest.fixture(scope="module", params=["title-ids"])
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "imports" / str(request.param)


def test_edges_match_snapshot(sample_edges: list[str], snapshot_json: SnapshotAssertion) -> None:
    assert sample_edges == snapshot_json
