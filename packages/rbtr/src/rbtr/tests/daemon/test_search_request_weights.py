"""Validation tests for `SearchRequest`'s weight override."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rbtr.config import WeightTriple
from rbtr.daemon.messages import SearchRequest

# ── default ──────────────────────────────────────────────────────────


def test_no_override_accepted() -> None:
    req = SearchRequest(repo_path="/r", query="q")
    assert req.weights is None


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
    wt = WeightTriple(alpha=a, beta=b, gamma=g)
    req = SearchRequest(repo_path="/r", query="q", weights=wt)
    assert req.weights is not None
    assert (req.weights.alpha, req.weights.beta, req.weights.gamma) == (a, b, g)


# ── range / sum ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "g"),
    [(-0.1, 0.5, 0.6), (0.4, 1.1, -0.5), (0.0, 0.0, 1.5)],
)
def test_out_of_range_rejected(a: float, b: float, g: float) -> None:
    with pytest.raises(ValidationError):
        WeightTriple(alpha=a, beta=b, gamma=g)


@pytest.mark.parametrize(
    ("a", "b", "g"),
    [(0.5, 0.5, 0.5), (0.0, 0.0, 0.5), (0.4, 0.4, 0.4)],
)
def test_sum_not_one_rejected(a: float, b: float, g: float) -> None:
    with pytest.raises(ValidationError):
        WeightTriple(alpha=a, beta=b, gamma=g)
