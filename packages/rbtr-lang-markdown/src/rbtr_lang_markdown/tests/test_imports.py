"""Markdown's links, each a small project under `samples/imports/`, built as rbtr builds a repository.

- `links`: A Markdown link documents what it names: a code file's symbols,
  or a document's sections.
- `link-to-a-heading`: A Markdown link to a heading's anchor documents that
  section alone.
- `renderer-anchors`: A link written with the anchor a renderer gives a
  heading documents it. The anchors are what the renderers themselves
  produced for the guide's headings: `github-slugger` 2.0.0 (GitHub;
  GitLab, Hugo and kramdown's GFM parser agree), Python-Markdown 3.11.3
  (MkDocs), Pandoc 3.13, and kramdown's own rule from its source.
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
        "link-to-a-heading",
        "renderer-anchors",
    ],
)
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "imports" / str(request.param)


def test_edges_match_snapshot(sample_edges: list[str], snapshot_json: SnapshotAssertion) -> None:
    assert sample_edges == snapshot_json
