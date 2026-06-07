"""Integration tests for the measure polars pipeline.

Exercises the composed shape `_run_* -> _score_* -> _aggregate`
against synthetic `SearchBatch` frames. No daemon, no embedding
model; the purpose is to catch schema drift and aggregation bugs
before the expensive indexing run hits them.

Row dicts are constructed inline inside the tests instead of
`pytest-cases` files because each behaviour needs its own
bespoke tiny dataset (2-6 rows); sharing would obscure the
scenario.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from rbtr_eval.measure import _aggregate, _score_outcomes
from rbtr_eval.schemas import SearchBatch


def _hit(file_path: str, scope: str, name: str, line_start: int = 1) -> dict[str, str | int]:
    """Build one hit-struct dict; helper keeps row literals readable."""
    return {"file_path": file_path, "scope": scope, "name": name, "line_start": line_start}


def _outcome_row(
    *,
    slug: str,
    target: str,
    latency_ms: float,
    hits: list[dict[str, str | int]],
    query_line_start: int = 1,
    language: str = "python",
    provenance: str = "docstring",
    arm: str = "none",
    query_kind: str = "concept",
) -> dict[str, str | int | float | list[dict[str, str | int]] | None]:
    """Build one `SearchBatch` row; `target` sets query_name."""
    return {
        "arm": arm,
        "slug": slug,
        "language": language,
        "query_file": "q.py",
        "query_scope": "",
        "query_name": target,
        "query_line_start": query_line_start,
        "provenance": provenance,
        "query_kind": query_kind,
        "query_text": f"doc of {target}",
        "latency_ms": latency_ms,
        "hits": hits,
        "expansion_kind": None,
        "expansion_n_keywords": None,
        "expansion_n_variants": None,
    }


# ── _score_outcomes ──────────────────────────────────────────────────────────


def test_score_outcomes_ranks_matching_hit_at_correct_position() -> None:
    """Target at position 2 in the hits list maps to `rank == 2`.

    The target query has `query_line_start=10`; only the hit
    at line 10 matches.  The hit at line 99 has the same
    `(file, scope, name)` but different line, testing the
    line-based disambiguation (e.g. property getters /
    setters at different lines with the same name).
    """
    batch = pl.DataFrame(
        [
            _outcome_row(
                slug="r",
                target="foo",
                latency_ms=1.0,
                query_line_start=10,
                hits=[
                    _hit("q.py", "", "foo", 99),
                    _hit("q.py", "", "foo", 10),
                ],
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    scored = _score_outcomes(batch)

    assert scored["rank"].to_list() == [2]
    assert scored["top_file"].to_list() == ["q.py"]
    assert scored["top_line"].to_list() == [99]
    assert scored["top_name"].to_list() == ["foo"]


def test_score_outcomes_gives_null_rank_when_target_missing() -> None:
    """Target absent from `hits` leaves `rank` null but top-1 still populated."""
    batch = pl.DataFrame(
        [
            _outcome_row(
                slug="r",
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

    Four queries under one slug: ranks 1, 3, null, null.
    `hit_at_1` is 1/4, `hit_at_3` is 2/4, `hit_at_10` is 2/4,
    MRR is (1 + 1/3 + 0 + 0) / 4 = 0.333...
    """
    batch = pl.DataFrame(
        [
            _outcome_row(slug="r", target="a", latency_ms=10.0, hits=[_hit("q.py", "", "a")]),
            _outcome_row(
                slug="r",
                target="b",
                latency_ms=20.0,
                hits=[_hit("x.py", "", "x"), _hit("y.py", "", "y"), _hit("q.py", "", "b")],
            ),
            _outcome_row(slug="r", target="c", latency_ms=30.0, hits=[_hit("x.py", "", "x")]),
            _outcome_row(slug="r", target="d", latency_ms=40.0, hits=[_hit("x.py", "", "x")]),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    metrics = _aggregate(_score_outcomes(batch))

    # Per-repo rollup row (language == '__all__', provenance == '__all__').
    per_repo = metrics.filter(
        (pl.col("slug") == "r")
        & (pl.col("language") == "__all__")
        & (pl.col("provenance") == "__all__")
    ).row(0, named=True)
    assert per_repo["n_queries"] == 4
    assert per_repo["hit_at_1"] == 0.25
    assert per_repo["hit_at_3"] == 0.5
    assert per_repo["hit_at_10"] == 0.5
    assert per_repo["mrr"] == (1 + 1 / 3) / 4
    assert per_repo["ndcg_at_10"] == pytest.approx(0.375)
    assert per_repo["not_found_pct"] == 0.5
    assert per_repo["median_rank"] == 2.0  # ranks in {1, 3}, median 2.0
    assert per_repo["search_p50_ms"] == 30.0
    assert per_repo["search_p95_ms"] == 40.0


def test_aggregate_emits_per_slug_plus_all_repos_rollup() -> None:
    """Two slugs produce 2 per-slug rows + 1 `__all__` rollup row."""
    batch = pl.DataFrame(
        [
            _outcome_row(slug="r1", target="a", latency_ms=1.0, hits=[_hit("q.py", "", "a")]),
            _outcome_row(slug="r2", target="b", latency_ms=3.0, hits=[_hit("q.py", "", "b")]),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    metrics = _aggregate(_score_outcomes(batch))

    assert sorted(metrics["slug"].unique().to_list()) == ["__all__", "r1", "r2"]


def test_score_outcomes_preserves_all_input_rows_even_when_target_misses() -> None:
    """Left-join shape: every input row appears exactly once in the output."""
    batch = pl.DataFrame(
        [
            _outcome_row(
                slug="r",
                target="hit",
                latency_ms=1.0,
                hits=[_hit("q.py", "", "hit")],
            ),
            _outcome_row(slug="r", target="miss", latency_ms=1.0, hits=[_hit("x", "", "x")]),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    scored = _score_outcomes(batch)

    assert scored.height == 2
    assert sorted(scored["rank"].to_list(), key=lambda x: (x is None, x)) == [1, None]
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
