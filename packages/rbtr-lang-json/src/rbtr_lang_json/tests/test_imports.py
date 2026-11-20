"""JSON references to other documents and their keys: `samples/imports/ref/`.

A `$ref` names a document and, after `#`, a JSON pointer into it; keys are
not scoped by their parents, so the pointer reaches the first key of its
last name. A pointer alone (`#/z`) is into the same document.
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
            "ref",
            {
                ("defs.json", "Named"): {
                    ("schema.json", '"$ref": "defs.json#/definitions/Named"', "imports")
                },
            },
            id="a-ref-reaches-the-key-its-pointer-names",
            marks=pytest.mark.xfail(
                strict=True,
                reason="a key that imports is one span captured twice, so the index keeps the key "
                "and loses the import (one-span-one-chunk Phase 5: a chunk lists its imports)",
            ),
        ),
        pytest.param(
            "ref",
            {
                ("schema.json", "z"): {("schema.json", '"$ref": "#/z"', "imports")},
            },
            id="a-pointer-alone-reaches-a-key-of-the-same-document",
            marks=pytest.mark.xfail(
                strict=True,
                reason="a key that imports is one span captured twice, so the index keeps the key "
                "and loses the import (one-span-one-chunk Phase 5: a chunk lists its imports)",
            ),
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
