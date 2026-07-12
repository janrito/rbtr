"""Cases for tune helper functions."""

from __future__ import annotations

import dataframely as dy
import polars as pl
from pytest_cases import case

from rbtr_eval.schemas import QueryMeta, ScoredCandidate

# ── _rescore_and_rank ────────────────────────────────────────────────────────


@case(tags=["rescore"])
def case_rescore_target_at_rank_one() -> tuple[
    dy.DataFrame[ScoredCandidate],
    dy.DataFrame[QueryMeta],
    tuple[float, float, float],
    list[int | None],
    float,
]:
    """Target has high semantic; semantic-heavy weights -> rank 1, MRR 1.0."""
    candidates = pl.DataFrame(
        [
            {
                "query_idx": 0,
                "file_path": "target.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
                "semantic": 0.9,
                "lexical": 0.1,
                "name_match": 0.1,
                "kind_boost": 1.0,
                "file_penalty": 1.0,
                "importance": 1.0,
                "proximity": 1.0,
            },
            {
                "query_idx": 0,
                "file_path": "other.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
                "semantic": 0.1,
                "lexical": 0.9,
                "name_match": 0.1,
                "kind_boost": 1.0,
                "file_penalty": 1.0,
                "importance": 1.0,
                "proximity": 1.0,
            },
        ]
    ).pipe(ScoredCandidate.validate, cast=True)
    meta = pl.DataFrame(
        [
            {
                "query_idx": 0,
                "slug": "repo",
                "language": "python",
                "provenance": "name",
                "query_kind": "identifier",
                "file_path": "target.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
            }
        ]
    ).pipe(QueryMeta.validate, cast=True)
    return candidates, meta, (0.8, 0.1, 0.1), [1], 1.0


@case(tags=["rescore"])
def case_rescore_target_drops_to_rank_two() -> tuple[
    dy.DataFrame[ScoredCandidate],
    dy.DataFrame[QueryMeta],
    tuple[float, float, float],
    list[int | None],
    float,
]:
    """Same candidates, lexical-heavy weights -> target drops to rank 2, MRR 0.5."""
    candidates = pl.DataFrame(
        [
            {
                "query_idx": 0,
                "file_path": "target.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
                "semantic": 0.9,
                "lexical": 0.1,
                "name_match": 0.1,
                "kind_boost": 1.0,
                "file_penalty": 1.0,
                "importance": 1.0,
                "proximity": 1.0,
            },
            {
                "query_idx": 0,
                "file_path": "other.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
                "semantic": 0.1,
                "lexical": 0.9,
                "name_match": 0.1,
                "kind_boost": 1.0,
                "file_penalty": 1.0,
                "importance": 1.0,
                "proximity": 1.0,
            },
        ]
    ).pipe(ScoredCandidate.validate, cast=True)
    meta = pl.DataFrame(
        [
            {
                "query_idx": 0,
                "slug": "repo",
                "language": "python",
                "provenance": "name",
                "query_kind": "identifier",
                "file_path": "target.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
            }
        ]
    ).pipe(QueryMeta.validate, cast=True)
    return candidates, meta, (0.1, 0.8, 0.1), [2], 0.5


@case(tags=["rescore"])
def case_rescore_target_outside_top_10() -> tuple[
    dy.DataFrame[ScoredCandidate],
    dy.DataFrame[QueryMeta],
    tuple[float, float, float],
    list[int | None],
    float,
]:
    """12 candidates; target ranks last -> filtered out -> null rank, near-zero MRR."""
    rows = [
        {
            "query_idx": 0,
            "file_path": f"c{i}.py",
            "scope": "",
            "name": "fn",
            "line_start": 1,
            "semantic": 0.8 - i * 0.05,
            "lexical": 0.8 - i * 0.05,
            "name_match": 0.5,
            "kind_boost": 1.0,
            "file_penalty": 1.0,
            "importance": 1.0,
            "proximity": 1.0,
        }
        for i in range(11)
    ]
    rows.append(
        {
            "query_idx": 0,
            "file_path": "target.py",
            "scope": "",
            "name": "fn",
            "line_start": 1,
            "semantic": 0.01,
            "lexical": 0.01,
            "name_match": 0.01,
            "kind_boost": 1.0,
            "file_penalty": 1.0,
            "importance": 1.0,
            "proximity": 1.0,
        }
    )
    candidates = pl.DataFrame(rows).pipe(ScoredCandidate.validate, cast=True)
    meta = pl.DataFrame(
        [
            {
                "query_idx": 0,
                "slug": "repo",
                "language": "python",
                "provenance": "name",
                "query_kind": "identifier",
                "file_path": "target.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
            }
        ]
    ).pipe(QueryMeta.validate, cast=True)
    # MRR = 0 for the only provenance; hmean floors to 1e-9.
    return candidates, meta, (0.4, 0.4, 0.2), [None], 1e-9


@case(tags=["rescore"])
def case_rescore_empty_candidates() -> tuple[
    dy.DataFrame[ScoredCandidate],
    dy.DataFrame[QueryMeta],
    tuple[float, float, float],
    list[int | None],
    float,
]:
    """Zero candidates for a query -> null rank, near-zero MRR."""
    candidates = pl.DataFrame(schema=ScoredCandidate.to_polars_schema()).pipe(
        ScoredCandidate.validate, cast=True
    )
    meta = pl.DataFrame(
        [
            {
                "query_idx": 0,
                "slug": "repo",
                "language": "python",
                "provenance": "name",
                "query_kind": "identifier",
                "file_path": "target.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
            }
        ]
    ).pipe(QueryMeta.validate, cast=True)
    # MRR = 0 for the only provenance; hmean floors to 1e-9.
    return candidates, meta, (0.5, 0.3, 0.2), [None], 1e-9


@case(tags=["rescore"])
def case_rescore_multiple_queries() -> tuple[
    dy.DataFrame[ScoredCandidate],
    dy.DataFrame[QueryMeta],
    tuple[float, float, float],
    list[int | None],
    float,
]:
    """Two queries, different provenances -> hmean of per-provenance MRRs.

    name provenance: rank 1 -> MRR 1.0
    body provenance: rank 3 -> MRR 1/3
    hmean = 2 / (1/1.0 + 1/(1/3)) = 0.5
    """
    candidates = pl.DataFrame(
        [
            # Query 0: target wins.
            {
                "query_idx": 0,
                "file_path": "target0.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
                "semantic": 0.9,
                "lexical": 0.9,
                "name_match": 0.5,
                "kind_boost": 1.0,
                "file_penalty": 1.0,
                "importance": 1.0,
                "proximity": 1.0,
            },
            {
                "query_idx": 0,
                "file_path": "other0.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
                "semantic": 0.1,
                "lexical": 0.1,
                "name_match": 0.5,
                "kind_boost": 1.0,
                "file_penalty": 1.0,
                "importance": 1.0,
                "proximity": 1.0,
            },
            # Query 1: target is 3rd of 3.
            {
                "query_idx": 1,
                "file_path": "other1a.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
                "semantic": 0.9,
                "lexical": 0.9,
                "name_match": 0.5,
                "kind_boost": 1.0,
                "file_penalty": 1.0,
                "importance": 1.0,
                "proximity": 1.0,
            },
            {
                "query_idx": 1,
                "file_path": "other1b.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
                "semantic": 0.8,
                "lexical": 0.8,
                "name_match": 0.5,
                "kind_boost": 1.0,
                "file_penalty": 1.0,
                "importance": 1.0,
                "proximity": 1.0,
            },
            {
                "query_idx": 1,
                "file_path": "target1.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
                "semantic": 0.1,
                "lexical": 0.1,
                "name_match": 0.5,
                "kind_boost": 1.0,
                "file_penalty": 1.0,
                "importance": 1.0,
                "proximity": 1.0,
            },
        ]
    ).pipe(ScoredCandidate.validate, cast=True)
    meta = pl.DataFrame(
        [
            {
                "query_idx": 0,
                "slug": "repo",
                "language": "python",
                "provenance": "name",
                "query_kind": "identifier",
                "file_path": "target0.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
            },
            {
                "query_idx": 1,
                "slug": "repo",
                "language": "python",
                "provenance": "body",
                "query_kind": "code",
                "file_path": "target1.py",
                "scope": "",
                "name": "fn",
                "line_start": 1,
            },
        ]
    ).pipe(QueryMeta.validate, cast=True)
    return candidates, meta, (0.5, 0.3, 0.2), [1, 3], 0.5
