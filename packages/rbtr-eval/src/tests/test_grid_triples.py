"""Tests for `tune.grid_triples` \u2014 pure projection per D13."""

from __future__ import annotations

import pytest

from rbtr_eval.tune import grid_triples


def test_step_one_returns_corners() -> None:
    assert sorted(grid_triples(1.0)) == sorted([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])


def test_step_half_returns_six_points() -> None:
    triples = grid_triples(0.5)
    assert len(triples) == 6
    for a, b, g in triples:
        assert abs(a + b + g - 1.0) < 1e-9
        assert 0.0 <= a <= 1.0
        assert 0.0 <= b <= 1.0
        assert 0.0 <= g <= 1.0


@pytest.mark.parametrize(
    ("step", "expected_count"),
    [
        (1.0, 3),
        (0.5, 6),
        (0.2, 21),
        (0.1, 66),
    ],
)
def test_count_matches_simplex_size(step: float, expected_count: int) -> None:
    triples = grid_triples(step)
    assert len(triples) == expected_count


def test_all_triples_unique() -> None:
    triples = grid_triples(0.2)
    assert len(triples) == len(set(triples))


@pytest.mark.parametrize("bad_step", [0.0, -0.1, 1.5])
def test_rejects_invalid_step(bad_step: float) -> None:
    with pytest.raises(ValueError, match="grid_step"):
        grid_triples(bad_step)
