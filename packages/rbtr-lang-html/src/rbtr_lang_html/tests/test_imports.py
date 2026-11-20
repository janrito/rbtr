"""HTML's script and stylesheet references, each a small project under `samples/imports/`, built as rbtr builds a repository.

- `scripts-and-a-stylesheet`: HTML resolves `<script src>` beside it and
  above it, and `<link href>` to a stylesheet.
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
            "scripts-and-a-stylesheet",
            {
                ("lib/core.js", "boot"): {
                    ("web/index.html", '<script src="../lib/core.js"></script>', "imports")
                },
                ("lib/core.js", "halt"): {
                    ("web/index.html", '<script src="../lib/core.js"></script>', "imports")
                },
                ("web/app.js", "start"): {
                    ("web/index.html", '<script src="./app.js"></script>', "imports")
                },
                ("web/styles.css", ".banner"): {
                    ("web/index.html", '<link rel="stylesheet" href="styles.css">', "imports")
                },
            },
            id="scripts-and-a-stylesheet",
        ),
    ],
)
def test_each_target_is_reached_by_exactly_the_chunks_named(
    indexed: Indexed,
    sample: str,
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target `(path, symbol)` of the sample under `samples/imports/` is reached by exactly the chunks named, by any edge."""
    with indexed(Path(__file__).parent / "samples" / "imports" / sample) as index:
        assert_reached(index, reached)
