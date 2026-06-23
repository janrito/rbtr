"""Unified search scoring and fusion.

Combines lexical (BM25), semantic (cosine), and name-match signals
into a single ranked result list.  Each signal is normalised to
[0, 1] via min-max scaling, then fused with a convex combination.
Post-fusion multipliers for chunk kind and file category adjust
the final score.

All scoring helpers are pure functions — easy to test in isolation.
Orchestration lives in `search()` which takes an `IndexStore` and
coordinates channel retrieval, candidate merging, and fusion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dataframely as dy
import duckdb
import polars as pl
import structlog

from rbtr.config import WeightTriple, config
from rbtr.index.classify import classify_query
from rbtr.index.frames import (
    _EMBEDDING_SENTINEL,
    ChunkPathResultRow,
    ChunkResultRow,
    EdgeResultRow,
    FusedRow,
    FusionInputRow,
    ScoredChunkResultRow,
)
from rbtr.index.models import ChunkKind, QueryKind, ScoredChunk, ScoredChunks

if TYPE_CHECKING:
    from rbtr.index.embeddings import Embedder
    from rbtr.index.models import RepoRef
    from rbtr.index.reranker import Reranker
    from rbtr.index.store import IndexStore

log = structlog.get_logger(__name__)

# Import and doc_section chunks produce misleadingly high cosine
# scores due to short, keyword-dense content.  Filtering them
# from the semantic pool lets real definitions surface.
_SEMANTIC_EXCLUDE = frozenset({ChunkKind.IMPORT, ChunkKind.DOC_SECTION})


def _filter_semantic(
    frame: pl.DataFrame,
    pool_size: int,
) -> dy.DataFrame[ScoredChunkResultRow]:
    """Remove excluded chunk kinds and limit to `pool_size`."""
    return ScoredChunkResultRow.cast(
        frame.filter(
            ~pl.col("kind").cast(pl.String).is_in([k.value for k in _SEMANTIC_EXCLUDE])
        ).head(pool_size)
    )


# ── Name matching ────────────────────────────────────────────────────


def _name_score_expr(query: str) -> pl.Expr:
    """Score how well *query* matches each chunk name.

    Operates on the `name` column, case-insensitive.  Checks
    whole-query matching first (exact > prefix > substring),
    then falls back to token-level matching: if every token in
    the query appears in the name, scores 0.4.  This handles
    multi-word concept queries like `"import edge"` matching
    `"infer_import_edges"`.

    Returns:
        1.0  exact match,
        0.8  prefix match,
        0.5  substring match,
        0.4  all query tokens found in name,
        0.0  no match.
    """
    query_lower = query.lower()
    if not query_lower:
        return pl.lit(0.0)

    name_lower = pl.col("name").str.to_lowercase()

    # Token-level: every query token must appear in name.
    tokens = query_lower.split()
    if len(tokens) > 1:
        all_tokens_found = name_lower.str.contains(tokens[0], literal=True)
        for token in tokens[1:]:
            all_tokens_found = all_tokens_found & name_lower.str.contains(token, literal=True)
    else:
        all_tokens_found = pl.lit(False)

    return (
        pl.when(name_lower == pl.lit(query_lower))
        .then(1.0)
        .when(name_lower.str.starts_with(query_lower))
        .then(0.8)
        .when(name_lower.str.contains(query_lower, literal=True))
        .then(0.5)
        .when(all_tokens_found)
        .then(0.4)
        .otherwise(0.0)
    )


# ── Query classification ─────────────────────────────────────────────


# ── Kind boost ───────────────────────────────────────────────────────


def _kind_boost_expr() -> pl.Expr:
    """Ranking boost multiplier based on chunk kind.

    Higher boosts for definitions (class, function, method),
    lower for noise (imports, raw chunks, migrations).
    Default 1.0 for unknown kinds.
    """
    kind = pl.col("kind")
    return (
        pl.when(kind == ChunkKind.CLASS)
        .then(1.5)
        .when(kind == ChunkKind.FUNCTION)
        .then(1.3)
        .when(kind == ChunkKind.METHOD)
        .then(1.3)
        .when(kind == ChunkKind.VARIABLE)
        .then(1.0)
        .when(kind == ChunkKind.API_ENDPOINT)
        .then(1.0)
        .when(kind == ChunkKind.DOC_SECTION)
        .then(0.8)
        .when(kind == ChunkKind.TEST_FUNCTION)
        .then(0.7)
        .when(kind == ChunkKind.CONFIG_KEY)
        .then(0.6)
        .when(kind == ChunkKind.RAW_CHUNK)
        .then(0.5)
        .when(kind == ChunkKind.MIGRATION)
        .then(0.4)
        .when(kind == ChunkKind.IMPORT)
        .then(0.3)
        .otherwise(1.0)
    )


# ── File category penalty ────────────────────────────────────────────


# Patterns for file classification.  Checked in order — first
# match wins.  Tuples of (substring_or_suffix, category).


def _file_category_penalty_expr() -> pl.Expr:
    """Ranking penalty based on file path category.

    Classifies by path patterns (checked in priority order):
    vendor (0.3) > generated (0.3) > test (0.5) > config (0.7)
    > doc (0.8) > source (1.0, default).
    """
    path = pl.col("file_path").str.to_lowercase()

    is_vendor = (
        path.str.contains("vendor/", literal=True)
        | path.str.contains("/vendor/", literal=True)
        | path.str.contains("node_modules/", literal=True)
        | path.str.contains("/node_modules/", literal=True)
        | path.str.contains("third_party/", literal=True)
        | path.str.contains("/third_party/", literal=True)
    )
    is_generated = (
        path.str.ends_with("_pb2.py")
        | path.str.ends_with(".pb.go")
        | path.str.contains("_generated.", literal=True)
        | path.str.contains(".gen.", literal=True)
    )
    is_test = (
        path.str.contains("test_", literal=True)
        | path.str.contains("/tests/", literal=True)
        | path.str.contains("_test.", literal=True)
        | path.str.contains("/test/", literal=True)
        | path.str.contains("tests.", literal=True)
        | path.str.contains("conftest.", literal=True)
    )
    is_doc = (
        path.str.ends_with(".md")
        | path.str.ends_with(".rst")
        | path.str.ends_with(".txt")
        | path.str.ends_with(".adoc")
    )
    is_config = (
        path.str.ends_with(".json")
        | path.str.ends_with(".yaml")
        | path.str.ends_with(".yml")
        | path.str.ends_with(".toml")
        | path.str.ends_with(".ini")
        | path.str.ends_with(".cfg")
    )

    return (
        pl.when(is_vendor)
        .then(0.3)
        .when(is_generated)
        .then(0.3)
        .when(is_test)
        .then(0.5)
        .when(is_doc)
        .then(0.8)
        .when(is_config)
        .then(0.7)
        .otherwise(1.0)
    )


# ── Importance (inbound-degree) ──────────────────────────────────────


def _importance_expr() -> pl.Expr:
    """Ranking boost from inbound edge count.

    Operates on a `degree` column.  Uses `log2(1 + degree)`
    scaled so that zero edges gives 1.0 (neutral) and higher
    degree gives a multiplicative boost.  Capped at 3.0.

    Examples:
        0 edges → 1.0  (no boost)
        1 edge  → 1.25
        3 edges → 1.50
        7 edges → 1.75
        15 edges → 2.00
    """
    degree = pl.col("degree").cast(pl.Float64)
    raw = 1.0 + (1 + degree).log(base=2) / 4
    return pl.when(degree <= 0).then(1.0).otherwise(pl.min_horizontal(raw, pl.lit(3.0)))


# ── Diff proximity ──────────────────────────────────────────────────


def compute_proximity(
    scored: pl.DataFrame,
    *,
    edge_frame: dy.DataFrame[EdgeResultRow],
    paths_frame: dy.DataFrame[ChunkPathResultRow],
    changed_files: set[str],
) -> pl.DataFrame:
    """Add a `prox` column ranking candidates by proximity to the diff.

    Returns plain `pl.DataFrame`: mid-pipeline transformation
    adding one column to an ad-hoc scored frame.

    Candidates in a changed file get 1.5.  Candidates with an
    edge (import, call, etc.) to a chunk in a changed file get
    1.2.  Candidates in the same directory as a changed file
    get 1.1.  All others get 1.0 (neutral).

    Args:
        scored:        Candidate frame with at least `id` and
                       `file_path` columns.
        edge_frame:    All edges for the commit (`source_id`,
                       `target_id`).  Used to find neighbours.
        paths_frame:   Chunk paths (`id`, `file_path`) covering
                       all IDs in *edge_frame*.  Used to resolve
                       neighbour file paths.
        changed_files: File paths modified in the current diff.
    """
    candidate_id_list = scored["id"].to_list()

    has_edge_list: list[str] = []
    if not edge_frame.is_empty():
        pairs = pl.concat(
            [
                edge_frame.select(
                    candidate_id=pl.col("source_id"), neighbour_id=pl.col("target_id")
                ).filter(pl.col("candidate_id").is_in(candidate_id_list)),
                edge_frame.select(
                    candidate_id=pl.col("target_id"), neighbour_id=pl.col("source_id")
                ).filter(pl.col("candidate_id").is_in(candidate_id_list)),
            ]
        )
        has_edge_list = (
            pairs.join(
                paths_frame.rename({"file_path": "nb_path"}),
                left_on="neighbour_id",
                right_on="id",
            )
            .filter(pl.col("nb_path").is_in(list(changed_files)))
            .select("candidate_id")
            .unique()
            .to_series()
            .to_list()
        )

    changed_dirs = sorted(cf.rsplit("/", 1)[0] for cf in changed_files if "/" in cf)

    return scored.with_columns(
        pl.when(pl.col("file_path").is_in(list(changed_files)))
        .then(1.5)
        .when(pl.col("id").is_in(has_edge_list))
        .then(1.2)
        .when(pl.col("file_path").str.replace(r"/[^/]+$", "").is_in(changed_dirs))
        .then(1.1)
        .otherwise(1.0)
        .alias("proximity"),
    )


def _normalise_col(col: str) -> pl.Expr:
    """Min-max normalise *col* to [0, 1] as a polars expression.

    Returns 0.0 when all values are equal (no signal).
    """
    c = pl.col(col)
    span = c.max() - c.min()
    return pl.when(span == 0.0).then(0.0).otherwise((c - c.min()) / span).alias(col)


def fuse_scores(
    scored: dy.DataFrame[FusionInputRow],
    query: str,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    top_k: int = 10,
) -> dy.DataFrame[FusedRow]:
    """Fuse three signal channels into a ranked result frame.

    Receives a `FusionInputRow` frame.  Computes name match
    scores, normalises all channels, applies kind boost and
    file penalty, and returns the top-k `FusedRow` frame.
    """
    if scored.is_empty():
        return FusedRow.create_empty()

    # From here on, `scored` gains computed columns beyond
    # FusionInputRow.  Work with pl.DataFrame internally.
    frame: pl.DataFrame = scored

    # ── Name score ───────────────────────────────────────────────
    frame = frame.with_columns(
        _name_score_expr(query).alias("name_match"),
    )

    # ── Normalise all three channels at once ─────────────────────
    frame = frame.with_columns(
        _normalise_col("lexical"),
        _normalise_col("semantic"),
        _normalise_col("name_match"),
    )

    # ── Kind boost + file category penalty ─────────────────────
    frame = frame.with_columns(
        _kind_boost_expr().alias("kind_boost"),
        _file_category_penalty_expr().alias("file_penalty"),
    )

    # ── Final score ──────────────────────────────────────────────
    frame = frame.with_columns(
        (
            (alpha * pl.col("semantic") + beta * pl.col("lexical") + gamma * pl.col("name_match"))
            * pl.col("kind_boost")
            * pl.col("file_penalty")
            * pl.col("importance")
            * pl.col("proximity")
        ).alias("score"),
    )

    # ── Top-k ───────────────────────────────────────────────────────
    return (
        frame.sort("score", "id", descending=[True, False])
        .head(top_k)
        .with_columns(
            pl.when(pl.col("has_embedding"))
            .then(pl.lit(_EMBEDDING_SENTINEL))
            .otherwise(pl.lit([]))
            .alias("embedding"),
            pl.col("score").alias("fusion"),
            pl.lit(0.0).alias("reranker"),
        )
        .drop("has_embedding")
        .pipe(FusedRow.validate, cast=True)
    )


def materialise_scored(
    frame: dy.DataFrame[FusedRow],
    repo_paths: dict[int, str] | None,
    query_kind: QueryKind,
) -> list[ScoredChunk]:
    """Serialise a fused frame to a list of `ScoredChunk`.

    *query_kind* is the classification that selected this search's
    weights; it is stamped onto every result (a query-level scalar
    that completes each chunk's score breakdown).

    When *repo_paths* maps `repo_id` to a filesystem path, each
    result's `repo_path` is populated from its `repo_id` column —
    used by cross-repo search to attribute results to their repo.
    When `None` (single-repo search), `repo_path` stays `None`.
    """
    if frame.is_empty():
        return []
    rows = frame.to_dicts()
    for row in rows:
        row["query_kind"] = query_kind
        if repo_paths is not None:
            row["repo_path"] = repo_paths.get(row["repo_id"])
    return ScoredChunks.validate_python(rows)


# ── Weight selection ─────────────────────────────────────────────────


def _select_reranker_params(
    kind: QueryKind,
    *,
    pool_override: int | None,
    blend_override: float | None,
) -> tuple[int, float]:
    """Resolve per-kind reranker pool and blend weight.

    Priority: explicit override > per-kind config default.
    """
    settings = config.reranker_settings[kind]
    pool = pool_override if pool_override is not None else settings.pool
    blend = blend_override if blend_override is not None else settings.blend_weight
    return pool, blend


def _select_weights(
    kind: QueryKind,
    override: WeightTriple | None,
    *,
    has_semantic: bool = True,
) -> tuple[float, float, float]:
    """Select and adjust fusion weights.

    Priority: explicit *override* > per-kind config.  When
    `has_semantic` is False the semantic weight is redistributed
    proportionally to the other two channels.
    """
    if override is not None:
        alpha, beta, gamma = override.alpha, override.beta, override.gamma
    else:
        t = config.search_weights[kind]
        alpha, beta, gamma = t.alpha, t.beta, t.gamma

    if not has_semantic and alpha > 0:
        total = beta + gamma
        beta = beta + alpha * (beta / total) if total > 0 else beta + alpha / 2
        gamma = gamma + alpha * (gamma / total) if total > 0 else gamma + alpha / 2
        alpha = 0.0

    return (alpha, beta, gamma)


# ── Unified search ───────────────────────────────────────────────────


def _embed_query(
    query: str,
    embedder: Embedder,
    variants: list[str] | None,
) -> list[list[float]]:
    """Embed the query and any expansion variants.

    Returns a list of vectors (original + variants), or an
    empty list if embedding fails.
    """
    try:
        prefix = config.query_instruction
        query_text = f"{prefix}{query}" if prefix else query
        original_vec = embedder.embed_single(query_text)

        variant_vecs: list[list[float]] = []
        if variants:
            try:
                variant_texts = [f"{prefix}{v}" if prefix else v for v in variants]
                variant_vecs = [r.vector for r in embedder.embed(variant_texts)]
            except (RuntimeError, ValueError):
                log.debug("variant_embedding_failed", exc_info=True)
    except (RuntimeError, ValueError):
        return []
    return [original_vec, *variant_vecs]


def _retrieve(
    store: IndexStore,
    refs: list[RepoRef],
    query: str,
    lex_query: str,
    query_vecs: list[list[float]],
    *,
    top_k: int = 10,
    changed_files: set[str] | None = None,
) -> dy.DataFrame[FusionInputRow]:
    """Three-channel retrieval + merge + importance + proximity.

    *refs* is the list of `(repo_id, commit_sha)` snapshots to
    search.  A single-repo search passes one ref; cross-repo
    search passes one per indexed repo and the channels span all
    of them in a single query each.

    `lex_query` is the BM25 query (already augmented with
    expansion keywords if applicable).  `query_vecs` are the
    pre-embedded query vectors for the semantic channel (empty
    to skip semantic).
    """
    # ── Channel 1: BM25 lexical ─────────────────────────────
    pool_size = top_k * config.retrieval_multiplier_lexical
    lex_frame = store._match_fulltext(refs, lex_query, top_k=pool_size)

    # ── Channel 2: semantic (embedding cosine) ───────────────
    sem_frame = ScoredChunkResultRow.create_empty()
    if query_vecs:
        try:
            sem_fetch = top_k * config.retrieval_multiplier_semantic
            raw_sem = store._match_similar(
                refs,
                query_vecs,
                sem_fetch,
            )
            sem_frame = _filter_semantic(raw_sem, pool_size)
        except duckdb.Error:
            pass

    # ── Channel 3: name match ───────────────────────────────
    name_frame = store._match_by_name(refs, query)

    # ── Merge candidates with scores via outer joins ─────────
    chunk_cols = list(ChunkResultRow.columns())
    scored = (
        lex_frame.rename({"score": "lexical"})
        .join(
            sem_frame.rename({"score": "semantic"}),
            on=chunk_cols,
            how="full",
            coalesce=True,
        )
        .join(name_frame, on=chunk_cols, how="full", coalesce=True)
        .with_columns(
            pl.col("lexical").fill_null(0.0),
            pl.col("semantic").fill_null(0.0),
        )
    )

    # ── Importance (inbound-degree) ──────────────────────
    candidate_ids = scored["id"].to_list()
    degree_frame = store.inbound_degrees(refs, candidate_ids)
    if not degree_frame.is_empty():
        imp = degree_frame.with_columns(
            _importance_expr().alias("importance"),
        ).select(pl.col("chunk_id").alias("id"), "importance")
        scored = scored.join(imp, on="id", how="left").with_columns(
            pl.col("importance").fill_null(1.0),
        )
    else:
        scored = scored.with_columns(pl.lit(1.0).alias("importance"))

    # ── Proximity (diff distance) ────────────────────────
    if changed_files:
        edge_frame = store._get_edges_frame(refs)
        # Build paths frame: candidates + unknown neighbours.
        edge_ids = pl.concat(
            [
                edge_frame.select(pl.col("source_id").alias("id")),
                edge_frame.select(pl.col("target_id").alias("id")),
            ]
        ).unique()
        unknown = edge_ids.join(scored.select("id"), on="id", how="anti")
        paths_frame = ChunkPathResultRow.cast(scored.select("id", "file_path"))
        unknown_ids = unknown["id"].to_list()
        if unknown_ids:
            extra = store._fetch_chunk_paths(refs, unknown_ids)
            paths_frame = ChunkPathResultRow.cast(pl.concat([paths_frame, extra]))

        scored = compute_proximity(
            scored,
            edge_frame=edge_frame,
            paths_frame=paths_frame,
            changed_files=changed_files,
        )
    else:
        scored = scored.with_columns(pl.lit(1.0).alias("proximity"))

    return scored.pipe(FusionInputRow.validate, cast=True)


def _has_semantic(candidates: dy.DataFrame[FusionInputRow]) -> bool:
    """Whether the semantic channel contributed to any candidate."""
    return not candidates.is_empty() and candidates["semantic"].max() != 0.0


def search(
    store: IndexStore,
    refs: list[RepoRef],
    query: str,
    *,
    top_k: int = 10,
    changed_files: set[str] | None = None,
    embedder: Embedder | None = None,
    kind: QueryKind | None = None,
    keywords: list[str] | None = None,
    variants: list[str] | None = None,
    weights: WeightTriple | None = None,
    reranker: Reranker | None = None,
    reranker_pool: int | None = None,
    reranker_blend_weight: float | None = None,
    repo_paths: dict[int, str] | None = None,
) -> list[ScoredChunk]:
    """Fused search combining lexical, semantic, and name signals.

    *refs* lists the `(repo_id, commit_sha)` snapshots to search
    across.  One ref is a single-repo search; many refs fan the
    query across repos and merge results into one ranked list.

    *repo_paths* maps `repo_id` to a filesystem path; when given,
    each result's `repo_path` is populated for cross-repo
    attribution.

    Classifies the query, selects fusion weights, prepares
    expansion inputs, delegates to `_retrieve` for
    three-channel retrieval, then to `fuse_scores` for
    normalisation, weighting, and top-k selection.
    """
    effective_kind = kind if kind is not None else classify_query(query)

    # ── Expansion: augment query for BM25 ────────────────────
    lex_query = query
    if keywords:
        lex_query = query + " " + " ".join(keywords)

    # ── Embed query + variants ─────────────────────────────
    query_vecs = _embed_query(query, embedder, variants) if embedder is not None else []

    candidates = _retrieve(
        store,
        refs,
        query,
        lex_query,
        query_vecs,
        top_k=top_k,
        changed_files=changed_files,
    )

    alpha, beta, gamma = _select_weights(
        effective_kind,
        weights,
        has_semantic=_has_semantic(candidates),
    )

    pool, blend = _select_reranker_params(
        effective_kind,
        pool_override=reranker_pool,
        blend_override=reranker_blend_weight,
    )
    fuse_top_k = max(pool, top_k) if reranker is not None else top_k
    frame = fuse_scores(
        candidates,
        query,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        top_k=fuse_top_k,
    )
    if reranker is not None:
        frame = reranker.rerank(query, frame, top_k=top_k, blend_weight=blend)
    return materialise_scored(frame, repo_paths, effective_kind)
