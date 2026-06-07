"""Query loading and sampling for rbtr-eval.

`load_all_queries` merges extract + concept parquets
into a single `dy.DataFrame[QueryRow]`. All downstream
stages use this instead of glob-reading parquets
themselves.

`subsample` is the single sampling strategy: cell-based
stratified sampling by `(slug, language, provenance)`.
Every consumer passes a different `queries_per_cell`
value.
"""

from __future__ import annotations

from pathlib import Path

import dataframely as dy
import polars as pl

from rbtr.index.classify import QueryKind
from rbtr_eval.schemas import QueryRow

PROVENANCE_TO_KIND: dict[str, str] = {
    "name": QueryKind.IDENTIFIER.value,
    "body": QueryKind.CODE.value,
    "docstring": QueryKind.CONCEPT.value,
    "concept": QueryKind.CONCEPT.value,
}


def load_all_queries(
    per_repo_dir: Path,
    concept_dir: Path,
) -> dy.DataFrame[QueryRow]:
    """Load and merge extract + concept queries.

    Reads all `*.parquet` files from both directories
    and returns a validated `QueryRow` frame.
    """
    query_files = sorted(per_repo_dir.glob("*.parquet"))
    concept_files = sorted(concept_dir.glob("*.parquet"))
    return pl.concat(
        [pl.read_parquet(f) for f in query_files + concept_files],
    ).pipe(QueryRow.validate, cast=True)


def subsample(
    queries: dy.DataFrame[QueryRow],
    *,
    queries_per_cell: int,
    seed: int,
    strat_keys: tuple[str, ...],
) -> dy.DataFrame[QueryRow]:
    """Cell-based stratified subsample.

    Samples up to `queries_per_cell` rows from each
    cell defined by `strat_keys`. Cells with fewer
    rows keep everything.
    """
    return (
        queries.group_by(*strat_keys)
        .map_groups(
            lambda g: g.sample(
                n=min(len(g), queries_per_cell),
                seed=seed,
                shuffle=False,
            )
        )
        .sort([*strat_keys, "file_path", "line_start", "scope", "name"])
        .pipe(QueryRow.validate, cast=True)
    )


def sample_distribution(
    queries: dy.DataFrame[QueryRow],
    strat_keys: tuple[str, ...],
) -> pl.DataFrame:
    """Query count per stratification cell.

    Returns a frame with one row per cell, columns
    `(*strat_keys, "n_queries")`, sorted by the keys.
    """
    return queries.group_by(*strat_keys).agg(pl.len().alias("n_queries")).sort(list(strat_keys))
