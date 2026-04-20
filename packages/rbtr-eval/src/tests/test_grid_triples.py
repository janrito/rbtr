"""Tests for `tune.grid_triples`."""

from __future__ import annotations

import pytest

from rbtr_eval.tune import grid_triples


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
    assert len(grid_triples(step)) == expected_count


def test_step_one_returns_only_corners() -> None:
    assert sorted(grid_triples(1.0)) == sorted(
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    )


@pytest.mark.parametrize("step", [1.0, 0.5, 0.2, 0.1])
def test_each_triple_lies_on_simplex(step: float) -> None:
    for a, b, g in grid_triples(step):
        assert abs(a + b + g - 1.0) < 1e-9
        assert 0.0 <= a <= 1.0
        assert 0.0 <= b <= 1.0
        assert 0.0 <= g <= 1.0


@pytest.mark.parametrize("step", [1.0, 0.5, 0.2, 0.1])
def test_triples_are_unique(step: float) -> None:
    triples = grid_triples(step)
    assert len(triples) == len(set(triples))


@pytest.mark.parametrize("bad_step", [0.0, -0.1, 1.5])
def test_rejects_invalid_step(bad_step: float) -> None:
    with pytest.raises(ValueError, match="grid_step"):
        grid_triples(bad_step)
