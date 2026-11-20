"""Which files the Rust plugin claims."""

from __future__ import annotations

import pytest

from rbtr.testing import Indexed
from rbtr.testing.reads import outline


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("lib.rs", "rust"),
    ],
)
def test_claims_its_files(indexed: Indexed, path: str, language: str) -> None:
    """A file at *path*, even an empty one, is read as *language*."""
    with indexed({path: ""}) as build:
        assert {symbol.language for symbol in outline(build)} == {language}
