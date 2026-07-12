"""Behaviour test for the reranker's latency-aware selection.

The report is a manual decision aid, so the load-bearing logic is
`_cheapest_within_tolerance`: as the MRR tolerance loosens it walks to
smaller — and therefore faster — pools. Latency is monotonic in pool.
"""

from __future__ import annotations

import polars as pl
import pytest

from rbtr_eval.tune_reranker import _cheapest_within_tolerance


@pytest.fixture
def reranker_configs() -> pl.DataFrame:
    """A (pool, blend) grid where a bigger pool is marginally better.

    Best MRR (0.90) is at pool 80; pool 50 is within 1% and pool 20
    within 3%, so the pick steps down to smaller pools as tolerance
    loosens. p50 latency rises with pool.
    """
    return pl.DataFrame(
        {
            "pool": [80, 50, 20, 80, 50, 20],
            "blend_weight": [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            "mrr": [0.90, 0.895, 0.88, 0.70, 0.69, 0.68],
            "search_p50_ms": [7000, 4800, 2100, 7000, 4800, 2100],
        }
    )


@pytest.mark.parametrize(
    ("tol", "expected_pool"),
    [(0.0, 80), (0.01, 50), (0.03, 20)],
)
def test_cheapest_within_tolerance_walks_to_faster_pools(
    reranker_configs: pl.DataFrame, tol: float, expected_pool: int
) -> None:
    """Looser MRR tolerance selects a smaller, faster pool."""
    pick = _cheapest_within_tolerance(reranker_configs, tol)
    assert pick["pool"] == expected_pool
    assert pick["blend_weight"] == 0.5  # best blend at the chosen pool
