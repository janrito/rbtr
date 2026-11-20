"""Which files the YAML plugin claims."""

from __future__ import annotations

import pytest

from rbtr.languages.manager import get_manager


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("config.yaml", "yaml"),
        ("ci.yml", "yaml"),
    ],
)
def test_claims_its_files(path: str, language: str) -> None:
    assert get_manager().detect_language(path) == language
