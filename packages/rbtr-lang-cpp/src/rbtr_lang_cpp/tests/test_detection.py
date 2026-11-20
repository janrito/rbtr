"""Which files the C++ plugin claims."""

from __future__ import annotations

import pytest

from rbtr.languages.manager import get_manager


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("order.cpp", "cpp"),
        ("order.hpp", "cpp"),
        ("kernels.cu", "cpp"),
        ("kernels.cuh", "cpp"),
    ],
)
def test_claims_its_files(path: str, language: str) -> None:
    assert get_manager().detect_language(path) == language
