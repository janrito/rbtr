"""Behaviour tests for `first_sentence`.

Thin assertions over case data: accepted docstrings project to
a known string; rejected ones project to `None`.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr_eval.extract import first_sentence


@parametrize_with_cases(
    "raw, expected",
    cases="tests.cases_first_sentence",
    has_tag="accepted",
)
def test_first_sentence_extracts_expected_text(raw: str, expected: str) -> None:
    """The first-sentence projection returns *expected* for *raw*."""
    assert first_sentence(raw) == expected


@parametrize_with_cases(
    "raw",
    cases="tests.cases_first_sentence",
    has_tag="rejected",
)
def test_first_sentence_rejects_docstring(raw: str) -> None:
    """Boilerplate, too-short, and noise-only docstrings yield None."""
    assert first_sentence(raw) is None
