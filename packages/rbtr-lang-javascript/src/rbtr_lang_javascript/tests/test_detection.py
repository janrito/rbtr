"""Which files the JavaScript plugin claims."""

from __future__ import annotations

import pytest

from rbtr.languages.manager import get_manager


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("index.js", "javascript"),
        ("App.jsx", "javascript"),
        ("module.mjs", "javascript"),
        ("app.ts", "typescript"),
        ("Component.tsx", "tsx"),
    ],
)
def test_claims_its_files(path: str, language: str) -> None:
    assert get_manager().detect_language(path) == language
