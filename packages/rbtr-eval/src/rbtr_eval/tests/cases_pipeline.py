"""Cases for the measure polars pipeline.

Each case returns a validated `SearchBatch` for one behaviour of the
`_score_outcomes |> _aggregate` chain. Rows are built with the `hit` /
`outcome_row` builders from `conftest`, so the case bodies stay
readable.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl
from pytest_cases import case

from rbtr_eval.schemas import SearchBatch

from .conftest import hit, outcome_row


@case(tags=["scoring"])
def case_ranks_and_top_hits() -> dy.DataFrame[SearchBatch]:
    """Three queries exercising every ranking behaviour.

    * `foo` has `query_line_start=10`; only the hit at line 10 matches,
      so the line-99 hit (same file/scope/name) does not, and the
      target lands at rank 2 — covering line-based disambiguation
      (e.g. getters / setters of the same name at different lines).
    * `absent` is missing from its hits — null rank, top-1 still set.
    * `found` sits first in its hits — rank 1.
    """
    return pl.DataFrame(
        [
            outcome_row(
                slug="r",
                target="foo",
                latency_ms=1.0,
                query_line_start=10,
                hits=[hit("q.py", "", "foo", 99), hit("q.py", "", "foo", 10)],
            ),
            outcome_row(
                slug="r",
                target="absent",
                latency_ms=1.0,
                hits=[hit("a.py", "", "bar"), hit("b.py", "", "baz")],
            ),
            outcome_row(
                slug="r",
                target="found",
                latency_ms=1.0,
                hits=[hit("q.py", "", "found")],
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)


@case(tags=["metrics"])
def case_known_rank_distribution() -> dy.DataFrame[SearchBatch]:
    """Five queries across two slugs with a known-good rank distribution.

    Slug `r` has ranks 1, 3, null, null and latencies 10/20/30/40;
    slug `r2` contributes one rank-1 query so the `__all__` rollup spans
    more than one repo.
    """
    return pl.DataFrame(
        [
            outcome_row(slug="r", target="a", latency_ms=10.0, hits=[hit("q.py", "", "a")]),
            outcome_row(
                slug="r",
                target="b",
                latency_ms=20.0,
                hits=[hit("x.py", "", "x"), hit("y.py", "", "y"), hit("q.py", "", "b")],
            ),
            outcome_row(slug="r", target="c", latency_ms=30.0, hits=[hit("x.py", "", "x")]),
            outcome_row(slug="r", target="d", latency_ms=40.0, hits=[hit("x.py", "", "x")]),
            outcome_row(slug="r2", target="e", latency_ms=5.0, hits=[hit("q.py", "", "e")]),
        ]
    ).pipe(SearchBatch.validate, cast=True)


@case(tags=["kinds"])
def case_two_kinds_two_shapes() -> dy.DataFrame[SearchBatch]:
    """Queries spanning two target kinds and two request shapes."""
    return pl.DataFrame(
        [
            outcome_row(
                slug="r",
                target="a",
                latency_ms=1.0,
                symbol_kind="function",
                query_kind="concept",
                hits=[hit("q.py", "", "a")],
            ),
            outcome_row(
                slug="r",
                target="b",
                latency_ms=1.0,
                symbol_kind="config_key",
                query_kind="identifier",
                hits=[hit("x.py", "", "x")],
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)
