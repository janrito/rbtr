"""Validation tests for `SearchRequest`'s alpha/beta/gamma override."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rbtr.daemon.messages import SearchRequest

# ── default ──────────────────────────────────────────────────────────


def test_no_override_accepted() -> None:
    req = SearchRequest(repo="/r", query="q")
    assert (req.alpha, req.beta, req.gamma) == (None, None, None)


# ── valid overrides ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "g"),
    [
        (0.0, 0.0, 1.0),
        (0.5, 0.3, 0.2),
        (1.0 / 3, 1.0 / 3, 1.0 / 3),
        (0.3333333, 0.3333333, 0.3333334),  # within 1e-6 tolerance
    ],
)
def test_valid_override_accepted(a: float, b: float, g: float) -> None:
    req = SearchRequest(repo="/r", query="q", alpha=a, beta=b, gamma=g)
    assert (req.alpha, req.beta, req.gamma) == (a, b, g)


# ── partial override ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "g"),
    [
        (0.5, 0.5, None),
        (0.5, None, 0.5),
        (None, 0.5, 0.5),
        (0.5, None, None),
    ],
)
def test_partial_override_rejected(a: float | None, b: float | None, g: float | None) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(repo="/r", query="q", alpha=a, beta=b, gamma=g)


# ── range / sum ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "g"),
    [(-0.1, 0.5, 0.6), (0.4, 1.1, -0.5), (0.0, 0.0, 1.5)],
)
def test_out_of_range_rejected(a: float, b: float, g: float) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(repo="/r", query="q", alpha=a, beta=b, gamma=g)


@pytest.mark.parametrize(
    ("a", "b", "g"),
    [(0.5, 0.5, 0.5), (0.0, 0.0, 0.5), (0.4, 0.4, 0.4)],
)
def test_sum_not_one_rejected(a: float, b: float, g: float) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(repo="/r", query="q", alpha=a, beta=b, gamma=g)
