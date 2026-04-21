"""Integration tests for the measure / tune polars pipeline.

Exercises the composed shape `_run_* -> _score_* -> _aggregate
-> _select_misses` against synthetic `SearchBatch` /
`WeightedSearchBatch` frames.  No daemon, no embedding model;
the purpose is to catch schema drift, pivot-name drift, and
aggregation bugs before the expensive indexing run hits them.

Row dicts are constructed inline inside the tests instead of
`pytest-cases` files because each behaviour needs its own
bespoke tiny dataset (2-6 rows); sharing would obscure the
scenario.
"""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from rbtr_eval.measure import _aggregate, _score_outcomes, _select_misses
from rbtr_eval.schemas import SearchBatch, WeightedSearchBatch
from rbtr_eval.tune import _score_trials


def _hit(file_path: str, scope: str, name: str, line_start: int = 1) -> dict[str, str | int]:
    """Build one hit-struct dict; helper keeps row literals readable."""
    return {"file_path": file_path, "scope": scope, "name": name, "line_start": line_start}


def _outcome_row(
    *,
    slug: str,
    variant: str,
    target: str,
    latency_ms: float,
    hits: list[dict[str, str | int]],
) -> dict[str, str | float | list[dict[str, str | int]]]:
    """Build one `SearchBatch` row; `target` sets query_name."""
    return {
        "slug": slug,
        "variant": variant,
        "query_file": "q.py",
        "query_scope": "",
        "query_name": target,
        "query_text": f"doc of {target}",
        "latency_ms": latency_ms,
        "hits": hits,
    }


# ── _score_outcomes ──────────────────────────────────────────────────────────


def test_score_outcomes_ranks_matching_hit_at_correct_position() -> None:
    """Target at position 2 in the hits list maps to `rank == 2`."""
    batch = pl.DataFrame(
        [
            _outcome_row(
                slug="r",
                variant="full",
                target="foo",
                latency_ms=1.0,
                hits=[_hit("a.py", "", "bar"), _hit("q.py", "", "foo", 10)],
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    scored = _score_outcomes(batch)

    assert scored["rank"].to_list() == [2]
    assert scored["top_file"].to_list() == ["a.py"]
    assert scored["top_name"].to_list() == ["bar"]


def test_score_outcomes_gives_null_rank_when_target_missing() -> None:
    """Target absent from `hits` leaves `rank` null but top-1 still populated."""
    batch = pl.DataFrame(
        [
            _outcome_row(
                slug="r",
                variant="full",
                target="missing",
                latency_ms=1.0,
                hits=[_hit("a.py", "", "bar"), _hit("b.py", "", "baz")],
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    scored = _score_outcomes(batch)

    assert scored["rank"].to_list() == [None]
    assert scored["top_file"].to_list() == ["a.py"]
    assert scored["top_name"].to_list() == ["bar"]


# ── _aggregate ───────────────────────────────────────────────────────────────


def test_aggregate_computes_hit_rates_and_mrr_with_null_ranks_as_misses() -> None:
    """Mixed ranks and one null compute Hit@k and MRR with null as zero reciprocal.

    Four queries under one `(slug, variant)` pair: ranks 1, 3,
    11 (i.e. 'would be 11, but clipped to null by the limit
    filter'), null.  The last two miss; `hit_at_1` is 1/4,
    `hit_at_3` is 2/4, `hit_at_10` is 2/4, MRR is
    (1 + 1/3 + 0 + 0) / 4 = 0.333...
    """
    batch = pl.DataFrame(
        [
            _outcome_row(
                slug="r", variant="full", target="a", latency_ms=10.0, hits=[_hit("q.py", "", "a")]
            ),
            _outcome_row(
                slug="r",
                variant="full",
                target="b",
                latency_ms=20.0,
                hits=[_hit("x.py", "", "x"), _hit("y.py", "", "y"), _hit("q.py", "", "b")],
            ),
            _outcome_row(
                slug="r", variant="full", target="c", latency_ms=30.0, hits=[_hit("x.py", "", "x")]
            ),
            _outcome_row(
                slug="r", variant="full", target="d", latency_ms=40.0, hits=[_hit("x.py", "", "x")]
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    metrics = _aggregate(_score_outcomes(batch))

    # Two rows: the per-pair + the __all__ rollup (same values, one variant).
    per_pair = metrics.filter(pl.col("slug") == "r").row(0, named=True)
    assert per_pair["n_queries"] == 4
    assert per_pair["hit_at_1"] == 0.25
    assert per_pair["hit_at_3"] == 0.5
    assert per_pair["hit_at_10"] == 0.5
    assert per_pair["mrr"] == (1 + 1 / 3) / 4
    assert per_pair["not_found_pct"] == 0.5
    assert per_pair["median_rank"] == 2.0  # ranks in {1, 3}, median 2.0
    # Polars `quantile` uses 'nearest' interpolation by default: on
    # `[10, 20, 30, 40]` the 50th-percentile index is
    # `round(1.5) = 2` -> `30.0`, not the linear midpoint `25.0`.
    assert per_pair["search_p50_ms"] == 30.0
    assert per_pair["search_p95_ms"] == 40.0


def test_aggregate_emits_per_pair_plus_all_repos_rollup() -> None:
    """Two slugs x two variants produce 4 per-pair rows + 2 `__all__` rollup rows."""
    batch = pl.DataFrame(
        [
            _outcome_row(
                slug="r1", variant="full", target="a", latency_ms=1.0, hits=[_hit("q.py", "", "a")]
            ),
            _outcome_row(
                slug="r1",
                variant="stripped",
                target="a",
                latency_ms=2.0,
                hits=[_hit("q.py", "", "a")],
            ),
            _outcome_row(
                slug="r2", variant="full", target="b", latency_ms=3.0, hits=[_hit("q.py", "", "b")]
            ),
            _outcome_row(
                slug="r2",
                variant="stripped",
                target="b",
                latency_ms=4.0,
                hits=[_hit("q.py", "", "b")],
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    metrics = _aggregate(_score_outcomes(batch))

    assert metrics.height == 6
    assert sorted(metrics["slug"].unique().to_list()) == ["__all__", "r1", "r2"]
    rollup_variants = sorted(
        metrics.filter(pl.col("slug") == "__all__")["variant"].cast(pl.String).to_list()
    )
    assert rollup_variants == ["full", "stripped"]


# ── _select_misses ───────────────────────────────────────────────────────────


def test_select_misses_keeps_only_queries_where_stripped_worse_than_full() -> None:
    """Gap column == rank_stripped - rank_full after sentinel substitution.

    Three queries: (1) stripped rank 5, full rank 1 -> gap 4,
    kept.  (2) stripped rank 1, full rank 3 -> gap -2, dropped.
    (3) both rank 2 -> gap 0, dropped.
    """
    batch = pl.DataFrame(
        [
            # Query `a`: full places it at rank 1, stripped at 5.
            _outcome_row(
                slug="r", variant="full", target="a", latency_ms=1.0, hits=[_hit("q.py", "", "a")]
            ),
            _outcome_row(
                slug="r",
                variant="stripped",
                target="a",
                latency_ms=1.0,
                hits=[
                    _hit("x", "", "x"),
                    _hit("y", "", "y"),
                    _hit("z", "", "z"),
                    _hit("w", "", "w"),
                    _hit("q.py", "", "a"),
                ],
            ),
            # Query `b`: full rank 3, stripped rank 1.
            _outcome_row(
                slug="r",
                variant="full",
                target="b",
                latency_ms=1.0,
                hits=[_hit("x", "", "x"), _hit("y", "", "y"), _hit("q.py", "", "b")],
            ),
            _outcome_row(
                slug="r",
                variant="stripped",
                target="b",
                latency_ms=1.0,
                hits=[_hit("q.py", "", "b")],
            ),
            # Query `c`: both rank 2.
            _outcome_row(
                slug="r",
                variant="full",
                target="c",
                latency_ms=1.0,
                hits=[_hit("x", "", "x"), _hit("q.py", "", "c")],
            ),
            _outcome_row(
                slug="r",
                variant="stripped",
                target="c",
                latency_ms=1.0,
                hits=[_hit("x", "", "x"), _hit("q.py", "", "c")],
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    misses = _select_misses(_score_outcomes(batch))

    assert misses["query_name"].to_list() == ["a"]
    assert misses["rank_full"].to_list() == [1]
    assert misses["rank_stripped"].to_list() == [5]
    assert misses["gap"].to_list() == [4]


def test_select_misses_sorts_by_gap_descending() -> None:
    """Several misses: largest gap first."""
    rows: list[dict[str, str | float | list[dict[str, str | int]]]] = []
    # Query `a`: full rank 1, stripped missing  -> gap 11 - 1 = 10.
    # Query `b`: full rank 1, stripped rank 3   -> gap 2.
    # Query `c`: full rank 2, stripped rank 8   -> gap 6.
    rows += [
        _outcome_row(
            slug="r", variant="full", target="a", latency_ms=1.0, hits=[_hit("q.py", "", "a")]
        ),
        _outcome_row(
            slug="r", variant="stripped", target="a", latency_ms=1.0, hits=[_hit("x", "", "x")]
        ),
        _outcome_row(
            slug="r", variant="full", target="b", latency_ms=1.0, hits=[_hit("q.py", "", "b")]
        ),
        _outcome_row(
            slug="r",
            variant="stripped",
            target="b",
            latency_ms=1.0,
            hits=[_hit("x", "", "x"), _hit("y", "", "y"), _hit("q.py", "", "b")],
        ),
        _outcome_row(
            slug="r",
            variant="full",
            target="c",
            latency_ms=1.0,
            hits=[_hit("x", "", "x"), _hit("q.py", "", "c")],
        ),
        _outcome_row(
            slug="r",
            variant="stripped",
            target="c",
            latency_ms=1.0,
            hits=[
                _hit("x", "", "x"),
                _hit("y", "", "y"),
                _hit("z", "", "z"),
                _hit("w", "", "w"),
                _hit("v", "", "v"),
                _hit("u", "", "u"),
                _hit("t", "", "t"),
                _hit("q.py", "", "c"),
            ],
        ),
    ]
    batch = pl.DataFrame(rows).pipe(SearchBatch.validate, cast=True)

    misses = _select_misses(_score_outcomes(batch))

    assert misses["query_name"].to_list() == ["a", "c", "b"]
    assert misses["gap"].to_list() == [10, 6, 2]


# ── tune: _score_trials ──────────────────────────────────────────────────────


def test_score_trials_ranks_each_weight_configuration_independently() -> None:
    """Baseline and grid rows for the same query get their own ranks.

    The query has its target at position 2 under baseline
    weights and position 1 under one grid triple.  Both
    outcomes appear in the scored frame with the right ranks.
    """
    batch = pl.DataFrame(
        [
            {
                "slug": "r",
                "label": "baseline",
                "query_file": "q.py",
                "query_scope": "",
                "query_name": "foo",
                "alpha": None,
                "beta": None,
                "gamma": None,
                "hits": [_hit("x", "", "x"), _hit("q.py", "", "foo")],
            },
            {
                "slug": "r",
                "label": "grid",
                "query_file": "q.py",
                "query_scope": "",
                "query_name": "foo",
                "alpha": 0.5,
                "beta": 0.5,
                "gamma": 0.0,
                "hits": [_hit("q.py", "", "foo"), _hit("x", "", "x")],
            },
        ]
    ).pipe(WeightedSearchBatch.validate, cast=True)

    scored = _score_trials(batch)

    assert scored.sort("label")["rank"].to_list() == [2, 1]


# ── shape sanity via assert_frame_equal ──────────────────────────────────────


def test_score_outcomes_preserves_all_input_rows_even_when_target_misses() -> None:
    """Left-join shape: every input row appears exactly once in the output."""
    batch = pl.DataFrame(
        [
            _outcome_row(
                slug="r",
                variant="full",
                target="hit",
                latency_ms=1.0,
                hits=[_hit("q.py", "", "hit")],
            ),
            _outcome_row(
                slug="r", variant="full", target="miss", latency_ms=1.0, hits=[_hit("x", "", "x")]
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    scored = _score_outcomes(batch)

    assert scored.height == 2
    # Ranks: one populated, one null.
    assert sorted(scored["rank"].to_list(), key=lambda x: (x is None, x)) == [1, None]
    # Expected frame (kept short - a fuller example is in the other tests).
    expected = pl.DataFrame(
        {
            "query_name": ["hit", "miss"],
            "rank": [1, None],
        },
        schema={"query_name": pl.String(), "rank": pl.UInt8()},
    )
    assert_frame_equal(
        scored.select("query_name", "rank").sort("query_name"), expected.sort("query_name")
    )
