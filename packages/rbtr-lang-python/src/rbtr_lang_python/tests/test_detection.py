"""Which files the Python plugin claims."""

from __future__ import annotations

import pytest

from rbtr.testing import Indexed
from rbtr.testing.reads import outline


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("src/app.py", "python"),
        ("stubs/foo.pyi", "python"),
        ("src/rbtr/index/store.py", "python"),
    ],
)
def test_claims_its_files(indexed: Indexed, path: str, language: str) -> None:
    """A file at *path*, even an empty one, is read as *language*."""
    with indexed({path: ""}) as index:
        assert {symbol.language for symbol in outline(index)} == {language}
