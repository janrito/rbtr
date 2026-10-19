"""Behaviour tests for the measure polars pipeline.

Exercises `_score_outcomes` and the statistics computed from it against
synthetic `SearchBatch` frames built in `cases_pipeline.py`. No daemon,
no embedding model; the purpose is to catch scoring and aggregation
bugs before the expensive indexing run hits them.
"""

from __future__ import annotations

from pathlib import Path

import dataframely as dy
import polars as pl
import pytest
from pytest_cases import parametrize_with_cases

from rbtr.index.index import Index
from rbtr_eval.measure import (
    SearchBatch,
    TruncatedTarget,
    _arm_metrics,
    _breakdowns,
    _metrics_file,
    _score_outcomes,
    _target_shapes,
    _truncated_targets,
)
from rbtr_eval.shared_schemas import RepoHeader
from rbtr_eval.tests.conftest import chunk, hit, outcome_row, snap


@parametrize_with_cases("batch", cases=".cases_pipeline", has_tag="scoring")
def test_scoring_assigns_ranks(batch: dy.DataFrame[SearchBatch]) -> None:
    """`_score_outcomes` ranks each target and preserves every input row.

    Covers hit position, line-based disambiguation, the null rank for a
    missing target (with top-1 still populated), and the left-join shape
    (every input row appears exactly once).
    """
    scored = _score_outcomes(batch, TruncatedTarget.create_empty()).sort("name")

    # Row preservation: every input row survives exactly once.
    assert scored.height == batch.height

    # Rank per target: found → 1, foo → 2 (line-disambiguated), absent → null.
    by_name = dict(zip(scored["name"], scored["rank"], strict=True))
    assert by_name == {"found": 1, "foo": 2, "absent": None}

    # Missing target keeps its top-1 fields populated.
    absent = scored.filter(pl.col("name") == "absent").row(0, named=True)
    assert absent["top_file"] == "a.py"
    assert absent["top_name"] == "bar"

    # Line disambiguation: the line-99 hit is top-1 even though the
    # rank-2 match is at line 10.
    foo = scored.filter(pl.col("name") == "foo").row(0, named=True)
    assert foo["top_file"] == "q.py"
    assert foo["top_line"] == 99
    assert foo["top_name"] == "foo"


@parametrize_with_cases("batch", cases=".cases_pipeline", has_tag="metrics")
def test_breakdowns_compute_metrics(batch: dy.DataFrame[SearchBatch]) -> None:
    """The per-repo breakdown computes Hit@k / MRR / NDCG and latency.

    Slug `r` has ranks 1, 3, null, null: Hit@1 = 1/4, Hit@3 = Hit@10 =
    2/4, MRR = (1 + 1/3 + 0 + 0) / 4, median over {1, 3} = 2.0, and the
    two null ranks count as misses (not_found = 0.5). Two slugs produce
    a row each.
    """
    by_repo = _breakdowns(_score_outcomes(batch, TruncatedTarget.create_empty())).filter(
        pl.col("dimension") == "slug"
    )

    per_repo = by_repo.filter(pl.col("value") == "r").row(0, named=True)
    assert per_repo["n_queries"] == 4
    assert per_repo["hit_at_1"] == 0.25
    assert per_repo["hit_at_3"] == 0.5
    assert per_repo["hit_at_10"] == 0.5
    assert per_repo["mrr"] == (1 + 1 / 3) / 4  # null ranks are zero reciprocal
    assert per_repo["ndcg_at_10"] == pytest.approx(0.375)
    assert per_repo["not_found_pct"] == 0.5
    assert per_repo["median_rank"] == 2.0
    assert per_repo["search_p50_ms"] == 30.0
    assert per_repo["search_p95_ms"] == 40.0

    assert by_repo["value"].to_list() == ["r", "r2"]


@parametrize_with_cases("batch", cases=".cases_pipeline", has_tag="kinds")
def test_statistics_break_out_symbol_kind(batch: dy.DataFrame[SearchBatch]) -> None:
    """Each target kind is measured on its own axis and in the cross
    with request shape."""
    outcomes = _score_outcomes(batch, TruncatedTarget.create_empty())

    per_kind = _breakdowns(outcomes).filter(pl.col("dimension") == "symbol_kind")
    assert set(per_kind["value"].to_list()) == {"function", "config_key"}

    cross = _target_shapes(outcomes)
    pairs = set(zip(cross["symbol_kind"].to_list(), cross["query_kind"].to_list(), strict=True))
    assert pairs == {("function", "concept"), ("config_key", "identifier")}


@pytest.fixture
def indexed_copy(tmp_path: Path) -> Path:
    """An index database holding one truncated chunk at two paths.

    `q.py` is vendored verbatim to `vendor/q.py`, so both snapshot
    rows carry the blob the chunk was extracted from.
    """
    index = Index(str(tmp_path / "index.duckdb"), writable=True)
    fn = chunk(file_path="q.py", kind="function", name="load")
    with index.session() as ws:
        repo = ws.register_repo(str(tmp_path / "alpha"))
        ws.add_chunk(fn)
        ws.insert_snapshots(
            [snap("a" * 40, fn), snap("a" * 40, fn, path="vendor/q.py")],
            repo=repo,
        )
        ws.mark_indexed(at=repo.at("a" * 40))
        ws.update_embeddings([fn.id], [[0.5] * 768], truncated=[True])
    index.close()
    return tmp_path / "index.duckdb"


def test_truncation_marks_every_location_of_the_chunk(indexed_copy: Path) -> None:
    """A truncated chunk is truncated wherever it is found.

    The measure joins on the path the query names, so a query against
    the vendored copy has to be marked from the same chunk row as one
    against the original.
    """
    batch = pl.DataFrame(
        [
            outcome_row(
                slug="alpha", target="load", latency_ms=1.0, hits=[hit("q.py", "", "load")]
            ),
            outcome_row(
                slug="alpha",
                target="load",
                latency_ms=1.0,
                file_path="vendor/q.py",
                hits=[hit("vendor/q.py", "", "load")],
            ),
            outcome_row(
                slug="alpha", target="other", latency_ms=1.0, hits=[hit("z.py", "", "other")]
            ),
        ]
    ).pipe(SearchBatch.validate, cast=True)

    annotated = _score_outcomes(batch, _truncated_targets(indexed_copy)).sort("file_path")

    marked = annotated.filter(pl.col("name") == "load")
    assert marked["file_path"].to_list() == ["q.py", "vendor/q.py"]
    assert marked["target_truncated"].to_list() == [True, True]

    unmarked = annotated.filter(pl.col("name") == "other")
    assert unmarked["target_truncated"].to_list() == [False]


@parametrize_with_cases("batch", cases=".cases_pipeline", has_tag="kinds")
def test_metrics_file_holds_each_arms_figures_overall_and_per_query_kind(
    batch: dy.DataFrame[SearchBatch], headers: dy.DataFrame[RepoHeader]
) -> None:
    """`dvc metrics` reads nested objects, so each figure sits under its arm,
    then `all` or the query kind."""
    outcomes = _score_outcomes(batch, TruncatedTarget.create_empty())

    metrics = _metrics_file(_arm_metrics(outcomes), _breakdowns(outcomes), headers, 1.0)

    assert set(metrics.arms) == {"none"}
    assert set(metrics.arms["none"]) == {"all", "concept", "identifier"}
    assert metrics.arms["none"]["all"]["n_queries"] == batch.height
    assert metrics.shas == {"r": "abc123def456"}
