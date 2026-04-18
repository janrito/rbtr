"""Behaviour tests for the bench query-sampler's `first_sentence`.

Thin assertions over case data: accepted docstrings project to
a known string; rejected ones project to `None`.
"""

from __future__ import annotations

from bench_docstrings import (
    first_sentence,  # type: ignore[import-not-found]  # sys.path set by conftest
)
from pytest_cases import parametrize_with_cases


@parametrize_with_cases(
    "raw, expected",
    cases="tests.scripts.case_first_sentence",
    has_tag="accepted",
)
def test_first_sentence_extracts_expected_text(raw: str, expected: str) -> None:
    """The first-sentence projection returns *expected* for *raw*."""
    assert first_sentence(raw) == expected


@parametrize_with_cases(
    "raw",
    cases="tests.scripts.case_first_sentence",
    has_tag="rejected",
)
def test_first_sentence_rejects_docstring(raw: str) -> None:
    """Boilerplate, too-short, and noise-only docstrings yield None."""
    assert first_sentence(raw) is None
