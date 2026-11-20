"""Which files the YAML plugin claims."""

from __future__ import annotations

import pytest

from rbtr.testing import Indexed
from rbtr.testing.reads import outline


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("config.yaml", "yaml"),
        ("ci.yml", "yaml"),
    ],
)
def test_claims_its_files(indexed: Indexed, path: str, language: str) -> None:
    """A file at *path*, even an empty one, is read as *language*."""
    with indexed({path: ""}) as index:
        assert {symbol.language for symbol in outline(index)} == {language}
