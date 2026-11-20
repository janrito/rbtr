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

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        pytest.param(
            "links",
            {
                ("docs/b.md", "Details"): {("docs/a.md", "b.md", "documents")},
                ("docs/b.md", "Guide"): {("docs/a.md", "b.md", "documents")},
                ("src/loader.py", "do_stuff"): {("docs/a.md", "../src/loader.py", "documents")},
                ("src/loader.py", "load"): {("docs/a.md", "../src/loader.py", "documents")},
            },
            id="links",
        ),
        pytest.param(
            "link-to-a-heading",
            {
                ("docs/b.md", "Details"): {("docs/a.md", "b.md#details", "documents")},
            },
            id="link-to-a-heading",
        ),
        pytest.param(
            "renderer-anchors",
            {
                ("docs/guide.md", "1. Numbered heading"): {
                    ("docs/readme.md", "guide.md#1-numbered-heading", "documents"),
                    ("docs/readme.md", "guide.md#numbered-heading", "documents"),
                },
                ("docs/guide.md", "API v2.0 — Overview"): {
                    ("docs/readme.md", "guide.md#api-v2.0-overview", "documents"),
                    ("docs/readme.md", "guide.md#api-v20--overview", "documents"),
                    ("docs/readme.md", "guide.md#api-v20-overview", "documents"),
                },
                ("docs/guide.md", "Already-hyphenated -- twice"): {
                    ("docs/readme.md", "guide.md#already-hyphenated----twice", "documents"),
                    ("docs/readme.md", "guide.md#already-hyphenated-twice", "documents"),
                },
                ("docs/guide.md", "C++ & C#"): {
                    ("docs/readme.md", "guide.md#c--c", "documents"),
                    ("docs/readme.md", "guide.md#c-c", "documents"),
                },
                ("docs/guide.md", "Café au lait"): {
                    ("docs/readme.md", "guide.md#cafe-au-lait", "documents"),
                    ("docs/readme.md", "guide.md#café-au-lait", "documents"),
                },
                ("docs/guide.md", "Duplicate"): {
                    ("docs/readme.md", "guide.md#duplicate", "documents")
                },
                ("docs/guide.md", "Emoji 🎉 party"): {
                    ("docs/readme.md", "guide.md#emoji--party", "documents"),
                    ("docs/readme.md", "guide.md#emoji-party", "documents"),
                },
                ("docs/guide.md", "Getting Started"): {
                    ("docs/readme.md", "guide.md#getting-started", "documents")
                },
                ("docs/guide.md", "Hello, World!"): {
                    ("docs/readme.md", "guide.md#hello-world", "documents")
                },
                ("docs/guide.md", "Multiple   spaces here"): {
                    ("docs/readme.md", "guide.md#multiple---spaces-here", "documents"),
                    ("docs/readme.md", "guide.md#multiple-spaces-here", "documents"),
                },
                ("docs/guide.md", "`code` in heading"): {
                    ("docs/readme.md", "guide.md#code-in-heading", "documents")
                },
                ("docs/guide.md", "snake_case_name"): {
                    ("docs/readme.md", "guide.md#snake_case_name", "documents"),
                    ("docs/readme.md", "guide.md#snakecasename", "documents"),
                },
                ("docs/guide.md", "Über Größe"): {
                    ("docs/readme.md", "guide.md#über-größe", "documents")
                },
                ("docs/guide.md", "日本語の見出し"): {
                    ("docs/readme.md", "guide.md#日本語の見出し", "documents")
                },
            },
            id="renderer-anchors",
        ),
    ],
)
def test_each_target_is_reached_by_exactly_the_chunks_named(
    indexed: Indexed,
    sample: str,
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target `(path, symbol)` of the sample under `samples/imports/` is reached by exactly the chunks named, by any edge."""
    with indexed(Path(__file__).parent / "samples" / "imports" / sample) as build:
        assert_reached(build, reached)
