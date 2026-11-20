"""JSON references to other documents and their keys: `samples/imports/ref/`.

A `$ref` names a document and, after `#`, a JSON pointer into it; keys are
not scoped by their parents, so the pointer reaches the first key of its
last name. A pointer alone (`#/z`) is into the same document.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sample_dir() -> Path:
    return Path(__file__).parent / "samples" / "imports" / "ref"


def test_a_ref_reaches_the_key_its_pointer_names(sample_edges: list[str]) -> None:
    assert (
        'schema.json::"$ref": "defs.json#/definitions/Named" -> defs.json::Named [imports]'
        in sample_edges
    )


def test_a_pointer_alone_reaches_a_key_of_the_same_document(sample_edges: list[str]) -> None:
    assert 'schema.json::"$ref": "#/z" -> schema.json::z [imports]' in sample_edges
