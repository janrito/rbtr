"""Dataframely schemas for every rbtr-eval data frame boundary.

Every function in `measure.py` / `tune.py` that takes or
returns a polars frame annotates the parameter / return as
`dy.DataFrame[Schema]` drawn from this module.  Frames are
validated at construction via
`pl.DataFrame(rows).pipe(Schema.validate, cast=True)`; the
schema is the one source of truth for the shape.
"""

from __future__ import annotations

from enum import StrEnum

import dataframely as dy
from pydantic import BaseModel, Field

from rbtr.index.models import ChunkKind
from rbtr_eval.kinds import EXCLUDED_KINDS


class ArmKind(StrEnum):
    """One expansion configuration measured per query.

    `measure` runs every query under each arm so the ablation
    can isolate the effect of each expansion channel.

    `NONE`     — no expansion (raw query only).
    `KEYWORDS` — keyword expansion only (lexical channel).
    `VARIANTS` — variant expansion only (semantic channel).
    `BOTH`     — both channels.
    """

    NONE = "none"
    KEYWORDS = "keywords"
    VARIANTS = "variants"
    BOTH = "both"


# Column shape of one hit inside the `hits: list[struct]`
# column on `SearchBatch`.  One entry per `ScoredResult`
# returned by the daemon; the ranking pipeline in `measure`
# explodes the list into per-hit rows and computes `rank` /
# `top_*` declaratively.
_HIT_COLUMNS: dict[str, dy.Column] = {
    "file_path": dy.String(),
    "scope": dy.String(),
    "name": dy.String(),
    "line_start": dy.UInt32(),
}


class QueryRow(dy.Schema):
    """One row per sampled query, emitted by `extract`.

    The per-repo `<slug>.parquet` file is the
    persisted form of this schema.  `measure` and `tune`
    read those files via `pl.read_parquet` + `QueryRow.validate`.
    `symbol_kind` is any `ChunkKind` the eval measures — every kind
    except `EXCLUDED_KINDS`.
    """

    slug = dy.String(primary_key=True)
    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    line_start = dy.UInt32(primary_key=True)
    symbol_kind = dy.Enum(k.value for k in ChunkKind if k not in EXCLUDED_KINDS)
    language = dy.String()
    provenance = dy.String(primary_key=True)
    text = dy.String()


class ExpansionRow(dy.Schema):
    """Pre-generated keywords and variants for a query."""

    slug = dy.String(primary_key=True)
    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    line_start = dy.UInt32(primary_key=True)
    provenance = dy.String(primary_key=True)
    query_kind = dy.String()
    keywords = dy.List(dy.String())
    variants = dy.List(dy.String())


class ConceptQuery(BaseModel):
    """LLM output: a one-sentence concept description."""

    text: str = Field(min_length=15, max_length=200)


class RepoHeader(dy.Schema):
    """One row per indexed repo.  Persisted in the headers directory.

    `sha` is the resolved HEAD SHA at extract time.  `seed` /
    `queries_per_cell` are stage parameters; `n_documented`
    is the total measurable-chunk count; `n_queries` is the
    post-subsample query count.  `dropped_languages` records the
    languages skipped for having fewer than `min_per_language`
    measurable chunks, with their chunk counts.
    """

    slug = dy.String(primary_key=True)
    sha = dy.String()
    seed = dy.UInt32()
    queries_per_cell = dy.UInt32(min=1)
    n_documented = dy.UInt32(min=0)
    n_queries = dy.UInt32(min=0)
    dropped_languages = dy.List(dy.Struct({"language": dy.String(), "n_chunks": dy.UInt32()}))


class SearchQuery(dy.Schema):
    """Identity and classification of one measured query.

    The `(arm, slug, language, query_file, query_scope, query_name,
    query_line_start, provenance)` key identifies the search; `arm` is
    the expansion configuration, `symbol_kind` the target chunk's kind,
    `query_kind` the `classify_query` shape.  `SearchBatch` and
    `SearchOutcome` extend this with the raw hits and the scored rank.
    """

    arm = dy.String(primary_key=True)
    slug = dy.String(primary_key=True)
    language = dy.String(primary_key=True)
    query_file = dy.String(primary_key=True)
    query_scope = dy.String(primary_key=True)
    query_name = dy.String(primary_key=True)
    query_line_start = dy.UInt32(primary_key=True)
    provenance = dy.String(primary_key=True)
    symbol_kind = dy.String()
    query_kind = dy.String()
    query_text = dy.String()
    latency_ms = dy.Float64(min=0.0)


class SearchBatch(SearchQuery):
    """Raw output of `_run_searches` before `_score_outcomes` runs.

    `hits` carries the top-10 results from the daemon as a
    list-of-struct; ranking turns this into `SearchOutcome` with
    scalar `rank` / `top_*` columns.
    """

    hits = dy.List(dy.Struct(_HIT_COLUMNS))
    expansion_kind = dy.String(nullable=True)
    expansion_n_keywords = dy.UInt8(nullable=True)
    expansion_n_variants = dy.UInt8(nullable=True)


class SearchOutcome(SearchQuery):
    """One scored search: the rank the target landed at and the top-1
    hit for diagnostics.  `rank` / `top_*` are null when the target
    does not appear in the top 10.
    """

    rank = dy.UInt8(nullable=True, min=1, max=10)
    top_file = dy.String(nullable=True)
    top_line = dy.UInt32(nullable=True)
    top_name = dy.String(nullable=True)
    target_truncated = dy.Bool()


class Metrics(dy.Schema):
    """Per-arm headline metrics plus rollups.

    Every level is partitioned by `arm` (always a real
    `ArmKind` value, never a sentinel).  Within an arm, the
    `slug` / `language` / `symbol_kind` / `provenance` /
    `query_kind` dimensions use the `'__all__'` sentinel for the
    dimensions a level does not span:

    * per group           → slug, language, provenance real
    * per repo+lang       → provenance == '__all__'
    * per language        → slug == '__all__', provenance == '__all__'
    * per provenance      → slug == '__all__', language == '__all__'
    * per symbol_kind      → slug/language/provenance == '__all__'
    * per query_kind      → slug/language/symbol_kind/provenance == '__all__'
    * per (symbol_kind, query_kind) → slug/language/provenance == '__all__'
    * global              → all '__all__'

    `median_rank` is null when every query missed.
    `search_p50_ms` / `search_p95_ms` come from the latency column
    on `SearchOutcome`.
    """

    arm = dy.String(primary_key=True)
    slug = dy.String(primary_key=True)
    language = dy.String(primary_key=True)
    symbol_kind = dy.String(primary_key=True)
    provenance = dy.String(primary_key=True)
    query_kind = dy.String(primary_key=True)
    n_queries = dy.UInt32(min=0)
    hit_at_1 = dy.Float64(min=0.0, max=1.0)
    hit_at_3 = dy.Float64(min=0.0, max=1.0)
    hit_at_10 = dy.Float64(min=0.0, max=1.0)
    mrr = dy.Float64(min=0.0, max=1.0)
    ndcg_at_10 = dy.Float64(min=0.0, max=1.0)
    median_rank = dy.Float64(nullable=True, min=1.0, max=10.0)
    not_found_pct = dy.Float64(min=0.0, max=1.0)
    search_p50_ms = dy.Float64(min=0.0)
    search_p95_ms = dy.Float64(min=0.0)


class MetricsFile(Metrics):
    """Shape of the on-disk `metrics.json` file.

    `Metrics` joined with per-slug SHA on `slug` (`__all__` rows stay
    null on `sha`), plus run metadata as literal columns so the JSON
    file carries everything DVC's metrics parser might want.
    """

    sha = dy.String(nullable=True)
    seed = dy.UInt32()
    queries_per_cell = dy.UInt32(min=1)
    elapsed_seconds = dy.Float64(min=0.0)


class ScoredCandidate(dy.Schema):
    """One candidate per query, carrying all component scores.

    Produced by `tune._collect_scored_candidates`; consumed
    by `tune._rescore_and_rank`.
    """

    query_idx = dy.UInt32()
    file_path = dy.String()
    scope = dy.String()
    name = dy.String()
    line_start = dy.UInt32()
    semantic = dy.Float64()
    lexical = dy.Float64()
    name_match = dy.Float64()
    kind_boost = dy.Float64()
    file_penalty = dy.Float64()
    importance = dy.Float64()
    proximity = dy.Float64()


class QueryMeta(dy.Schema):
    """Query identity columns, indexed by `query_idx`.

    Produced alongside the scored-candidate frame by
    `tune._collect_scored_candidates` and
    `tune_reranker._collect_candidates`.  `query_kind` is the
    request classification from `classify_query(text)` — the same
    axis production routes on.
    """

    query_idx = dy.UInt32()
    slug = dy.String()
    language = dy.String()
    provenance = dy.String()
    query_kind = dy.String()
    file_path = dy.String()
    scope = dy.String()
    name = dy.String()
    line_start = dy.UInt32()


class DetailedOutcome(dy.Schema):
    """Per-query rank from a single weight configuration.

    Produced by `tune._rescore_and_rank`; consumed by
    `tune._impact_comparison`.
    """

    slug = dy.String()
    language = dy.String()
    provenance = dy.String()
    rank = dy.UInt8(nullable=True, min=1, max=10)


class RerankerCandidate(dy.Schema):
    """One row per (pool, query, result) from the daemon.

    Produced by `tune_reranker._collect_candidates`;
    consumed by `tune_reranker._rank_all_blends`.
    """

    pool = dy.Int64(min=1)
    query_idx = dy.UInt32()
    file_path = dy.String()
    scope = dy.String()
    name = dy.String()
    line_start = dy.UInt32()
    fusion = dy.Float64()
    reranker = dy.Float64()
    latency_ms = dy.Float64(min=0.0)


class ImpactComparison(dy.Schema):
    """Side-by-side MRR for baseline vs best weights.

    One row per rollup dimension (repo, language, provenance,
    and `__all__` sentinels for rollups).
    """

    slug = dy.String()
    language = dy.String()
    provenance = dy.String()
    baseline_mrr = dy.Float64(min=0.0, max=1.0)
    best_mrr = dy.Float64(min=0.0, max=1.0)
    delta = dy.Float64(min=-1.0, max=1.0)
    baseline_ndcg_at_10 = dy.Float64(min=0.0, max=1.0)
    best_ndcg_at_10 = dy.Float64(min=0.0, max=1.0)
    delta_ndcg_at_10 = dy.Float64(min=-1.0, max=1.0)


class TuneReport(dy.Schema):
    """Shape of the on-disk `tuned-params.json` file.

    One row per `QueryKind` (concept, identifier, code).
    """

    kind = dy.String()
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
    n_trials = dy.UInt32(min=1)
    n_queries = dy.UInt32(min=0)
    elapsed_seconds = dy.Float64(min=0.0)
