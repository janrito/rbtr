"""rST's links, each a small project under `samples/imports/`, built as rbtr builds a repository.

- `title-ids`: A hyperlink written with the id docutils gives a section's
  title documents that section. The ids are what docutils 0.23 itself
  gave the guide's titles.
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
            "title-ids",
            {
                ("docs/guide.rst", "1. Numbered heading"): {
                    (
                        "docs/index.rst",
                        "`numbered-heading <guide.rst#numbered-heading>`_",
                        "documents",
                    )
                },
                ("docs/guide.rst", "API v2.0 — Overview"): {
                    (
                        "docs/index.rst",
                        "`api-v2-0-overview <guide.rst#api-v2-0-overview>`_",
                        "documents",
                    )
                },
                ("docs/guide.rst", "Already-hyphenated -- twice"): {
                    (
                        "docs/index.rst",
                        "`already-hyphenated-twice <guide.rst#already-hyphenated-twice>`_",
                        "documents",
                    )
                },
                ("docs/guide.rst", "C++ & C#"): {
                    ("docs/index.rst", "`c-c <guide.rst#c-c>`_", "documents")
                },
                ("docs/guide.rst", "Café au lait"): {
                    ("docs/index.rst", "`cafe-au-lait <guide.rst#cafe-au-lait>`_", "documents")
                },
                ("docs/guide.rst", "Duplicate"): {
                    ("docs/index.rst", "`duplicate <guide.rst#duplicate>`_", "documents")
                },
                ("docs/guide.rst", "Emoji 🎉 party"): {
                    ("docs/index.rst", "`emoji-party <guide.rst#emoji-party>`_", "documents")
                },
                ("docs/guide.rst", "Getting Started"): {
                    (
                        "docs/index.rst",
                        "`getting-started <guide.rst#getting-started>`_",
                        "documents",
                    )
                },
                ("docs/guide.rst", "Hello, World!"): {
                    ("docs/index.rst", "`hello-world <guide.rst#hello-world>`_", "documents")
                },
                ("docs/guide.rst", "Multiple   spaces here"): {
                    (
                        "docs/index.rst",
                        "`multiple-spaces-here <guide.rst#multiple-spaces-here>`_",
                        "documents",
                    )
                },
                ("docs/guide.rst", "snake_case_name"): {
                    (
                        "docs/index.rst",
                        "`snake-case-name <guide.rst#snake-case-name>`_",
                        "documents",
                    )
                },
            },
            id="title-ids",
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
