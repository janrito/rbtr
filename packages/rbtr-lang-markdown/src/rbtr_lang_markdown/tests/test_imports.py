"""Markdown's links, each a small project under `samples/imports/`, built as rbtr builds a repository.

- `links`: A Markdown link documents what it names: a code file's symbols,
  or a document's sections.
- `link-to-a-heading`: A Markdown link to a heading's anchor documents that
  section alone.
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
        "links",
        pytest.param(
            "link-to-a-heading",
            marks=pytest.mark.xfail(
                reason="a link names a heading by its anchor (`#details`), a section by its text",
                strict=True,
            ),
        ),
    ],
)
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "imports" / str(request.param)


def test_edges_match_snapshot(sample_edges: list[str], snapshot_json: SnapshotAssertion) -> None:
    assert sample_edges == snapshot_json
