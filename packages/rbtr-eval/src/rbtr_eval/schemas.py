"""Dataframely schemas for every rbtr-eval data frame boundary.

Every function in `measure.py` / `tune.py` that takes or
returns a polars frame annotates the parameter / return as
`dy.DataFrame[Schema]` drawn from this module.  Frames are
validated at construction via
`pl.DataFrame(rows).pipe(Schema.validate, cast=True)`; the
schema is the one source of truth for the shape.
"""

from __future__ import annotations

import dataframely as dy

from rbtr.index.models import ChunkKind, IndexVariant

# Column shape of one hit inside the `hits: list[struct]`
# column on `SearchBatch` / `WeightedSearchBatch`.  One entry
# per `ScoredResult` returned by the daemon; the ranking
# pipelines in `measure` and `tune` explode the list into
# per-hit rows and compute `rank` / `top_*` declaratively.
# `line_start` is only consumed by `measure`'s misses
# appendix; tune ignores it.
_HIT_COLUMNS: dict[str, dy.Column] = {
    "file_path": dy.String(),
    "scope": dy.String(),
    "name": dy.String(),
    "line_start": dy.UInt32(),
}


class QueryRow(dy.Schema):
    """One row per sampled query, emitted by `extract`.

    The per-repo `<slug>.queries.parquet` file is the
    persisted form of this schema.  `measure` and `tune`
    read those files via `pl.read_parquet` + `QueryRow.validate`.
    `symbol_kind` is narrowed to the three `ChunkKind` values
    rbtr-eval samples; the filter on the extract side must
    stay in sync with this whitelist.
    """

    slug = dy.String(primary_key=True)
    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    symbol_kind = dy.Enum(k.value for k in (ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD))
    line_start = dy.UInt32()
    language = dy.String()
    text = dy.String()


class RepoHeader(dy.Schema):
    """One row per indexed repo.  Persisted as `<slug>.header.parquet`.

    `sha` is the resolved HEAD SHA at extract time.  `seed` /
    `sample_cap` are stage parameters; `n_documented` /
    `n_sampled` are the outcome of sampling.
    """

    slug = dy.String(primary_key=True)
    sha = dy.String()
    seed = dy.UInt32()
    sample_cap = dy.UInt32(min=1)
    n_documented = dy.UInt32(min=0)
    n_sampled = dy.UInt32(min=0)


class SearchOutcome(dy.Schema):
    """One row per `(slug, variant, query)` search executed by `measure`.

    Records the rank the target landed at, the wall-clock
    latency of the search, and the top-1 hit's location for
    the misses appendix.  `rank` / `top_*` are nullable when
    the target does not appear in the top 10.
    """

    slug = dy.String(primary_key=True)
    variant = dy.Enum([v.value for v in IndexVariant], primary_key=True)
    query_file = dy.String(primary_key=True)
    query_scope = dy.String(primary_key=True)
    query_name = dy.String(primary_key=True)
    query_text = dy.String()
    rank = dy.UInt8(nullable=True, min=1, max=10)
    latency_ms = dy.Float64(min=0.0)
    top_file = dy.String(nullable=True)
    top_line = dy.UInt32(nullable=True)
    top_name = dy.String(nullable=True)


class SearchBatch(dy.Schema):
    """Raw output of `_run_searches` before `_score_outcomes` runs.

    One row per `(slug, variant, query)` search.  `hits`
    carries the top-10 results from the daemon as a
    list-of-struct; ranking turns this into `SearchOutcome`
    with scalar `rank` / `top_*` columns.
    """

    slug = dy.String(primary_key=True)
    variant = dy.Enum([v.value for v in IndexVariant], primary_key=True)
    query_file = dy.String(primary_key=True)
    query_scope = dy.String(primary_key=True)
    query_name = dy.String(primary_key=True)
    query_text = dy.String()
    latency_ms = dy.Float64(min=0.0)
    hits = dy.List(dy.Struct(_HIT_COLUMNS))


class WeightedSearchOutcome(dy.Schema):
    """One row per `(slug, label, triple, query)` search executed by `tune`.

    `label` is `"baseline"` when rbtr's configured defaults
    apply (`alpha` / `beta` / `gamma` are null) or `"grid"`
    when a specific triple is being trialled.  No primary
    key is declared because baseline rows have null triples
    (dataframely forbids nullable PK components) and grid
    rows need the triple for uniqueness; the two labels'
    uniqueness contracts diverge, so the schema guards
    column shape only.
    """

    slug = dy.String()
    label = dy.Enum(["baseline", "grid"])
    query_file = dy.String()
    query_scope = dy.String()
    query_name = dy.String()
    alpha = dy.Float64(nullable=True, min=0.0, max=1.0)
    beta = dy.Float64(nullable=True, min=0.0, max=1.0)
    gamma = dy.Float64(nullable=True, min=0.0, max=1.0)
    rank = dy.UInt8(nullable=True, min=1, max=10)


class WeightedSearchBatch(dy.Schema):
    """Raw output of `_run_weight_trials` before `_score_trials` runs.

    Same shape as `WeightedSearchOutcome` minus `rank`, plus
    a `hits: list[struct]` column.  Like `WeightedSearchOutcome`,
    no primary key — the baseline / grid labels have
    divergent uniqueness contracts.
    """

    slug = dy.String()
    label = dy.Enum(["baseline", "grid"])
    query_file = dy.String()
    query_scope = dy.String()
    query_name = dy.String()
    alpha = dy.Float64(nullable=True, min=0.0, max=1.0)
    beta = dy.Float64(nullable=True, min=0.0, max=1.0)
    gamma = dy.Float64(nullable=True, min=0.0, max=1.0)
    hits = dy.List(dy.Struct(_HIT_COLUMNS))


class Metrics(dy.Schema):
    """Per-(slug, variant) headline metrics plus `slug == '__all__'` rollup rows.

    `median_rank` is null for a variant in which every query
    missed (polars' `drop_nulls().median()` on an empty series
    returns null).  `search_p50_ms` / `search_p95_ms` come
    from the latency column on `SearchOutcome`.
    """

    slug = dy.String(primary_key=True)
    variant = dy.Enum([v.value for v in IndexVariant], primary_key=True)
    n_queries = dy.UInt32(min=0)
    hit_at_1 = dy.Float64(min=0.0, max=1.0)
    hit_at_3 = dy.Float64(min=0.0, max=1.0)
    hit_at_10 = dy.Float64(min=0.0, max=1.0)
    mrr = dy.Float64(min=0.0, max=1.0)
    median_rank = dy.Float64(nullable=True, min=1.0, max=10.0)
    not_found_pct = dy.Float64(min=0.0, max=1.0)
    search_p50_ms = dy.Float64(min=0.0)
    search_p95_ms = dy.Float64(min=0.0)


class MissCandidate(dy.Schema):
    """One row per query where `stripped` did worse than `full`.

    Produced by `df.pivot(on='variant', ...)` of the
    `SearchOutcome` frame; pivot column-name convention is
    `<values>_<variant>` so each search-outcome column appears
    twice with suffixes.  `gap` is
    `rank_stripped - rank_full`, with missing ranks
    substituted by 11 (sentinel for "worse than any real
    rank").
    """

    slug = dy.String(primary_key=True)
    query_file = dy.String(primary_key=True)
    query_scope = dy.String(primary_key=True)
    query_name = dy.String(primary_key=True)
    query_text = dy.String()
    rank_full = dy.UInt8(nullable=True, min=1, max=10)
    rank_stripped = dy.UInt8(nullable=True, min=1, max=10)
    top_file_full = dy.String(nullable=True)
    top_file_stripped = dy.String(nullable=True)
    top_line_full = dy.UInt32(nullable=True)
    top_line_stripped = dy.UInt32(nullable=True)
    top_name_full = dy.String(nullable=True)
    top_name_stripped = dy.String(nullable=True)
    gap = dy.Int16()


class MetricsFile(dy.Schema):
    """Shape of the on-disk `metrics.json` file.

    `Metrics` columns, joined with per-slug SHA on `slug`
    (`__all__` rows stay null on `sha`), with run metadata
    repeated as literal columns so the JSON file carries
    everything DVC's metrics parser might want.
    """

    slug = dy.String(primary_key=True)
    variant = dy.Enum([v.value for v in IndexVariant], primary_key=True)
    n_queries = dy.UInt32(min=0)
    hit_at_1 = dy.Float64(min=0.0, max=1.0)
    hit_at_3 = dy.Float64(min=0.0, max=1.0)
    hit_at_10 = dy.Float64(min=0.0, max=1.0)
    mrr = dy.Float64(min=0.0, max=1.0)
    median_rank = dy.Float64(nullable=True, min=1.0, max=10.0)
    not_found_pct = dy.Float64(min=0.0, max=1.0)
    search_p50_ms = dy.Float64(min=0.0)
    search_p95_ms = dy.Float64(min=0.0)
    sha = dy.String(nullable=True)
    rbtr_sha = dy.String()
    seed = dy.UInt32()
    sample_cap = dy.UInt32(min=1)
    elapsed_seconds = dy.Float64(min=0.0)
    index_size_bytes = dy.UInt64(min=0)


class TuneReport(dy.Schema):
    """Shape of the on-disk `tuned-params.json` file (one row)."""

    best_alpha = dy.Float64(min=0.0, max=1.0)
    best_beta = dy.Float64(min=0.0, max=1.0)
    best_gamma = dy.Float64(min=0.0, max=1.0)
    score_best = dy.Float64(min=0.0, max=1.0)
    current_alpha = dy.Float64(min=0.0, max=1.0)
    current_beta = dy.Float64(min=0.0, max=1.0)
    current_gamma = dy.Float64(min=0.0, max=1.0)
    score_current = dy.Float64(min=0.0, max=1.0)
    delta = dy.Float64(min=-1.0, max=1.0)
    metric = dy.String()
    grid_step = dy.Float64(min=0.0, max=1.0)
    n_queries = dy.UInt32(min=0)
    n_grid_points = dy.UInt32(min=1)
    rbtr_sha = dy.String()
    elapsed_seconds = dy.Float64(min=0.0)
