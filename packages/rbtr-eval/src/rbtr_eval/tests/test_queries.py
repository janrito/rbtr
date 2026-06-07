"""Behaviour tests for `queries.subsample` and `queries.load_all_queries`.

`subsample(queries, queries_per_cell, seed, strat_keys)` deterministically
subsamples a `QueryRow` frame stratified by `(slug, language,
provenance)`.  The observable behaviours:

- cells already at or below the target pass through unchanged,
- larger cells are capped to the target,
- results are deterministic across calls with the same seed,
- output validates as `QueryRow`.

`load_all_queries` merges extract + concept parquets into one
validated frame.
"""

from __future__ import annotations

from pathlib import Path

import dataframely as dy
import polars as pl
import pytest

from rbtr_eval.queries import load_all_queries, subsample
from rbtr_eval.schemas import QueryRow


@pytest.fixture
def queries() -> dy.DataFrame[QueryRow]:
    """Synthetic QueryRow frame with two repos and three provenances.

    Sizes chosen so name and body are well above a typical cap
    while docstring is naturally scarce.
    """
    rows = []
    for slug, lang in [("repo_a", "python"), ("repo_b", "rust")]:
        for kind, n in [("name", 80), ("body", 60), ("docstring", 10)]:
            for i in range(n):
                rows.append(
                    {
                        "slug": slug,
                        "file_path": f"{lang}/{kind}_{i}.py",
                        "scope": "",
                        "name": f"fn_{kind}_{i}",
                        "line_start": i + 1,
                        "symbol_kind": "function",
                        "language": lang,
                        "provenance": kind,
                        "text": f"query {slug} {kind} {i}",
                    }
                )
    return pl.DataFrame(rows).pipe(QueryRow.validate, cast=True)


@pytest.mark.parametrize(
    ("queries_per_cell", "expected_docstring_n"),
    [
        (50, 10),  # cap above docstring cell size
        (100, 10),  # cap above all cell sizes — full passthrough
    ],
    ids=["partial-cap", "full-passthrough"],
)
def test_small_cells_kept_whole(
    queries: dy.DataFrame[QueryRow],
    queries_per_cell: int,
    expected_docstring_n: int,
) -> None:
    """Cells with fewer rows than the target keep all rows."""
    result = subsample(
        queries,
        queries_per_cell=queries_per_cell,
        seed=0,
        strat_keys=("slug", "language", "provenance"),
    )
    # docstring cells have 10 rows each — should survive intact.
    docstring = result.filter(pl.col("provenance") == "docstring")
    per_slug = dict(docstring.group_by("slug").agg(pl.len().alias("n")).iter_rows())
    assert per_slug["repo_a"] == expected_docstring_n
    assert per_slug["repo_b"] == expected_docstring_n
    # Full passthrough: every row survives.
    if queries_per_cell >= 100:
        assert result.height == queries.height


def test_large_cells_capped(queries: dy.DataFrame[QueryRow]) -> None:
    """Cells larger than the target are capped."""
    cap = 30
    result = subsample(
        queries,
        queries_per_cell=cap,
        seed=0,
        strat_keys=("slug", "language", "provenance"),
    )
    counts = result.group_by("slug", "language", "provenance").agg(pl.len().alias("n"))
    for row in counts.iter_rows(named=True):
        assert row["n"] <= cap


def test_deterministic(queries: dy.DataFrame[QueryRow]) -> None:
    """Same seed produces identical output."""
    a = subsample(
        queries,
        queries_per_cell=30,
        seed=42,
        strat_keys=("slug", "language", "provenance"),
    )
    b = subsample(
        queries,
        queries_per_cell=30,
        seed=42,
        strat_keys=("slug", "language", "provenance"),
    )
    assert a.equals(b)


def test_different_seeds_differ(queries: dy.DataFrame[QueryRow]) -> None:
    """Different seeds produce different samples."""
    a = subsample(
        queries,
        queries_per_cell=30,
        seed=0,
        strat_keys=("slug", "language", "provenance"),
    )
    b = subsample(
        queries,
        queries_per_cell=30,
        seed=99,
        strat_keys=("slug", "language", "provenance"),
    )
    assert not a.equals(b)


def test_validates_as_query_row(queries: dy.DataFrame[QueryRow]) -> None:
    """Subsampled frame passes QueryRow schema validation."""
    result = subsample(
        queries,
        queries_per_cell=30,
        seed=0,
        strat_keys=("slug", "language", "provenance"),
    )
    result.pipe(QueryRow.validate, cast=True)


def test_all_cells_present(queries: dy.DataFrame[QueryRow]) -> None:
    """Every (slug, language, provenance) cell present in the input survives."""
    cap = 30
    result = subsample(
        queries,
        queries_per_cell=cap,
        seed=0,
        strat_keys=("slug", "language", "provenance"),
    )
    cell_keys = ["slug", "language", "provenance"]
    original_cells = set(queries.select(cell_keys).unique().iter_rows())
    result_cells = set(result.select(cell_keys).unique().iter_rows())
    assert result_cells == original_cells


def test_load_all_queries(tmp_path: Path) -> None:
    """Merges extract + concept parquets into one validated frame."""
    per_repo_dir = tmp_path / "per-repo"
    per_repo_dir.mkdir()
    concept_dir = tmp_path / "concept"
    concept_dir.mkdir()

    base_row = {
        "scope": "",
        "symbol_kind": "function",
        "language": "python",
    }

    # Two extract parquets.
    extract_a = pl.DataFrame(
        [
            {
                **base_row,
                "slug": "repo_a",
                "file_path": "a.py",
                "name": "fn_a",
                "line_start": 1,
                "provenance": "name",
                "text": "fn_a",
            },
        ]
    )
    extract_b = pl.DataFrame(
        [
            {
                **base_row,
                "slug": "repo_b",
                "file_path": "b.py",
                "name": "fn_b",
                "line_start": 1,
                "provenance": "docstring",
                "text": "does something",
            },
        ]
    )
    extract_a.write_parquet(per_repo_dir / "repo_a.parquet")
    extract_b.write_parquet(per_repo_dir / "repo_b.parquet")

    # One concept parquet.
    concept = pl.DataFrame(
        [
            {
                **base_row,
                "slug": "repo_a",
                "file_path": "a.py",
                "name": "fn_a",
                "line_start": 1,
                "provenance": "concept",
                "text": "a concept query",
            },
        ]
    )
    concept.write_parquet(concept_dir / "repo_a.parquet")

    result = load_all_queries(per_repo_dir, concept_dir)

    # Validates as QueryRow (load_all_queries does this internally).
    result.pipe(QueryRow.validate, cast=True)
    # Row count = sum of extract + concept rows.
    assert result.height == 3
    # All provenances present.
    provenances = set(result["provenance"].to_list())
    assert provenances == {"name", "docstring", "concept"}
