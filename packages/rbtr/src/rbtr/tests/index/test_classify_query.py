"""Tests for `classify_query` — the heuristic query classifier.

Correct-classification cases are tagged `identifier`, `concept`,
or `code` in `cases_classify_query.py`.  Known misclassifications
are tagged `misclassified` and run as `xfail` — if the heuristic
improves and one starts passing, pytest flags it as XPASS.
"""

from __future__ import annotations

import pytest
from pytest_cases import get_case_tags, parametrize_with_cases

from rbtr.index.classify import QueryKind, _code_score, classify_query

# ── Classification ───────────────────────────────────────────────────


@parametrize_with_cases(
    "query, expected_kind",
    filter=lambda c: "misclassified" not in get_case_tags(c),
)
def test_classifies_correctly(query: str, expected_kind: QueryKind) -> None:
    assert classify_query(query) == expected_kind


@pytest.mark.xfail(reason="Known heuristic trade-off", strict=True)
@parametrize_with_cases("query, expected_kind", has_tag="misclassified")
def test_known_misclassifications(query: str, expected_kind: QueryKind) -> None:
    assert classify_query(query) == expected_kind


# ── Scoring ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("query", "expected_score"),
    [
        ("fuse_scores", 0),
        ("search results", 0),
        # Keyword at start only (1 pt).
        ("import os", 1),
        ("class Foo", 1),
        # Trailing colon only (1 pt).
        ("engines:", 1),
        # Keyword + trailing colon (2 pts).
        ("class Engine:", 2),
        # Stopword + trailing colon (1 pt only).
        ("for item in self._cache:", 1),
        # Function call (2 pts).
        ("foo(bar)", 2),
        # Braces (2 pts).
        ("body { color: red }", 2),
        # Operators (2 pts).
        ("a == b", 2),
        ("a => b", 2),
        # Angle bracket pair (2 pts).
        ("Result<T>", 2),
        # Stacked signals.
        ("function update() {", 5),
        ("def fuse_scores(", 3),
    ],
)
def test_code_score(query: str, expected_score: int) -> None:
    assert _code_score(query) == expected_score
