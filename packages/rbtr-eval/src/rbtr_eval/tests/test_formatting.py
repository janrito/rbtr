"""Tests for shared eval-report formatting helpers."""

from __future__ import annotations

import pytest

from rbtr_eval.formatting import heading_label

_MULTILINE_SELECTOR = """\
.change-list .filtered .results,
.change-list .filtered .paginator,
.filtered #toolbar,
.filtered .actions,
#changelist .paginator"""


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("findPosX", "findPosX"),
        ("  spaced  ", "spaced"),
        ("td\n\nth", "td …"),
        (_MULTILINE_SELECTOR, ".change-list .filtered .results, …"),
        ("\n\n", ""),
        ("", ""),
    ],
)
def test_heading_label(name: str, expected: str) -> None:
    assert heading_label(name) == expected
