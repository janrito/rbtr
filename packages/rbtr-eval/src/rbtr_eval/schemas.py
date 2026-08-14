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

from rbtr.domain.models import ChunkKind
from rbtr_eval.kinds import EXCLUDED_KINDS

IDENTITY_COLUMNS: tuple[str, ...] = (
    "file_path",
    "scope",
    "name",
    "line_start",
    "line_end",
    "symbol_kind",
)
"""Identifies one target chunk. Two chunks can start on one line, so
the span needs both ends and the kind."""

_CHUNK_KIND_VALUES: tuple[str, ...] = tuple(k.value for k in ChunkKind)
"""Categories of every `symbol_kind` enum in this module.

A `pl.Enum` dtype is identified by its category list *in order*, so two
enums built from differently ordered categories are different dtypes and
refuse to join.  These categories are therefore taken from `ChunkKind` in
declaration order, matching `rbtr.index.results.ChunkContentRow.kind`, and
frames read from the index store join eval frames without a cast.  Sorting,
filtering or otherwise reordering this tuple breaks that silently, at the
join rather than here.
"""

_EXCLUDED_KIND_VALUES: tuple[str, ...] = tuple(k.value for k in EXCLUDED_KINDS)
"""Kinds the eval generates no queries for; excluded from target columns."""


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
    # Every location the hit's content sits at: identical content is one
    # chunk, so a hit reaches the target file when it is one of these.
    "file_paths": dy.List(dy.String()),
    "scope": dy.String(),
    "name": dy.String(),
    "line_start": dy.UInt32(),
    "line_end": dy.UInt32(),
    "symbol_kind": dy.Enum(_CHUNK_KIND_VALUES),
}


class QueryRow(dy.Schema):
    """One row per sampled query, emitted by `extract`.

    The per-repo `<slug>.parquet` file is the
    persisted form of this schema.  `measure` and `tune`
    read those files via `pl.read_parquet` + `QueryRow.validate`.
    `symbol_kind` spans the whole `ChunkKind` domain, so a target frame
    joins one read from the index store; a `check` restricts the values
    to the kinds the eval measures — every kind except `EXCLUDED_KINDS`.

    The key is `IDENTITY_COLUMNS` within a `slug`, plus the
    `provenance` that generated the text: one target yields at most one
    name, one body and one docstring query.
    """

    slug = dy.String(primary_key=True)
    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    line_start = dy.UInt32(primary_key=True)
    line_end = dy.UInt32(primary_key=True)
    symbol_kind = dy.Enum(
        _CHUNK_KIND_VALUES,
        primary_key=True,
        check=lambda kind: ~kind.is_in(_EXCLUDED_KIND_VALUES),
    )
    language = dy.String()
    provenance = dy.String(primary_key=True)
    text = dy.String()


class ExpansionRow(dy.Schema):
    """Pre-generated keywords and variants for a query.

    Keyed like `QueryRow`, so `measure` joins expansions onto queries on
    the target they were generated for.
    """

    slug = dy.String(primary_key=True)
    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    line_start = dy.UInt32(primary_key=True)
    line_end = dy.UInt32(primary_key=True)
    symbol_kind = dy.Enum(_CHUNK_KIND_VALUES, primary_key=True)
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

    The search is identified by `arm`, `slug`, the target
    (`IDENTITY_COLUMNS`, under those same names) and `provenance`; `arm`
    is the expansion configuration and `query_kind` the
    `classify_query` shape.  `SearchBatch` and `SearchOutcome` extend
    this with the raw hits and the scored rank, and a hit's own columns
    carry a `hit_` prefix so the two never collide.
    """

    arm = dy.String(primary_key=True)
    slug = dy.String(primary_key=True)
    language = dy.String(primary_key=True)
    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    line_start = dy.UInt32(primary_key=True)
    line_end = dy.UInt32(primary_key=True)
    symbol_kind = dy.Enum(_CHUNK_KIND_VALUES, primary_key=True)
    provenance = dy.String(primary_key=True)
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
    # A grouping dimension, not a target's kind: it holds a `ChunkKind`
    # or the sentinel standing for every kind at once, so it is a
    # different column from the `symbol_kind` the other schemas carry
    # and does not share their enum.
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

    `file_paths` holds every location the candidate's content sits at,
    as the daemon returns it, so a target is reached when its path is
    one of them.
    """

    query_idx = dy.UInt32()
    file_paths = dy.List(dy.String())
    scope = dy.String()
    name = dy.String()
    line_start = dy.UInt32()
    line_end = dy.UInt32()
    symbol_kind = dy.Enum(_CHUNK_KIND_VALUES)
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
    line_end = dy.UInt32()
    symbol_kind = dy.Enum(_CHUNK_KIND_VALUES)


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
    file_paths = dy.List(dy.String())
    scope = dy.String()
    name = dy.String()
    line_start = dy.UInt32()
    line_end = dy.UInt32()
    symbol_kind = dy.Enum(_CHUNK_KIND_VALUES)
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


class RepoCountRow(dy.Schema):
    """One row of INDEX.md's per-repo table.

    Counts are over the repo's corpus snapshot only (see
    `rbtr_eval.corpus`). `chunks` counts content, so a file vendored
    twice contributes one chunk and two `locations`, and two repos
    holding the same file each count it. `locations` is what compares
    against a file count.
    """

    repo = dy.String()
    chunks = dy.UInt64(min=0)
    locations = dy.UInt64(min=0)
    edges = dy.UInt64(min=0)


class KindCountRow(dy.Schema):
    """One row of INDEX.md's per-kind table.

    An edge contributes to `outbound_edges` of its source kind and
    `inbound_edges` of its target kind, so both columns sum to the
    edge total.
    """

    kind = dy.Enum(k.value for k in ChunkKind)
    n = dy.UInt64(min=0)
    outbound_edges = dy.UInt64(min=0)
    inbound_edges = dy.UInt64(min=0)


class LanguageCountRow(dy.Schema):
    """One row of INDEX.md's per-language table.

    `lang` is the plugin's language id, or `(plaintext)` for chunks
    from files no plugin claimed.
    """

    lang = dy.String()
    n = dy.UInt64(min=0)
    outbound_edges = dy.UInt64(min=0)
    inbound_edges = dy.UInt64(min=0)


class EmbeddingRepoRow(dy.Schema):
    """One row of EMBEDDING.md's per-repo table.

    `embedded` counts chunks carrying a vector and `truncated` those
    whose text exceeded the model's context, so
    `embedded <= chunks` holds by construction.
    """

    repo = dy.String()
    chunks = dy.UInt64(min=0)
    embedded = dy.UInt64(min=0)
    truncated = dy.UInt64(min=0)
