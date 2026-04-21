"""Behaviour tests for `tune.grid_triples`.

`grid_triples(step)` enumerates `(alpha, beta, gamma)` on the
unit simplex at the given resolution.  The observable
behaviours are:

- the right number of points for a given step,
- every point is on the simplex (sum == 1, each component
  in [0, 1]),
- points are unique,
- step=1 returns exactly the three corners,
- bad steps raise.
"""

from __future__ import annotations

import pytest

from rbtr_eval.tune import grid_triples

# Every step for which we check invariants.  One place so new
# resolutions get covered by every invariant test automatically.
_STEPS_AND_COUNTS = [(1.0, 3), (0.5, 6), (0.2, 21), (0.1, 66)]
_STEPS = [s for s, _ in _STEPS_AND_COUNTS]


@pytest.mark.parametrize(("step", "expected_count"), _STEPS_AND_COUNTS)
def test_count_matches_simplex_size(step: float, expected_count: int) -> None:
    """Count at resolution *step* equals the triangular number for the simplex."""
    assert len(grid_triples(step)) == expected_count


def test_step_one_returns_only_corners() -> None:
    """At the coarsest resolution the three triples are the simplex corners."""
    assert sorted(grid_triples(1.0)) == sorted([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])


@pytest.mark.parametrize("step", _STEPS)
def test_triples_sum_to_one(step: float) -> None:
    """Every returned triple sums to 1 within float tolerance."""
    assert all(abs(a + b + g - 1.0) < 1e-9 for a, b, g in grid_triples(step))


@pytest.mark.parametrize("step", _STEPS)
def test_triples_components_in_unit_range(step: float) -> None:
    """Every component of every triple is in `[0, 1]`."""
    assert all(0.0 <= x <= 1.0 for triple in grid_triples(step) for x in triple)


@pytest.mark.parametrize("step", _STEPS)
def test_triples_are_unique(step: float) -> None:
    """No duplicate triples at any resolution."""
    triples = grid_triples(step)
    assert len(triples) == len(set(triples))


@pytest.mark.parametrize("bad_step", [0.0, -0.1, 1.5])
def test_rejects_invalid_step(bad_step: float) -> None:
    """Out-of-range steps raise ValueError mentioning `grid_step`."""
    with pytest.raises(ValueError, match="grid_step"):
        grid_triples(bad_step)
