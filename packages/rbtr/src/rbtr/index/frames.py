"""Polars DataFrame builders for bulk-loading index data into DuckDB.

DuckDB's `executemany` has ~1 ms/row overhead.  Registering a
frame as a virtual view and running `INSERT INTO ... SELECT`
against it is orders of magnitude faster for large batches.
`duckdb.register` accepts polars frames natively (shared Arrow
memory, zero-copy), so the builders return
`dy.DataFrame[Schema]` instead of hand-rolled `pa.Table`.

Each builder converts a list of domain objects (`Chunk`,
`Edge`, or snapshot tuples) into a typed polars frame whose
column names match the corresponding SQL staging view
(`_stg`).  Validation via dataframely catches schema drift at
the function boundary.

These are pure functions -- they never touch DuckDB directly.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl

from rbtr.index.models import (
    ChangeKind,
    Chunk,
    ChunkKind,
    Chunks,
    Edge,
    EdgeKind,
    Edges,
    ImportMeta,
    RepoRef,
    Snapshot,
    Snapshots,
    TokenisedChunk,
    TokenisedChunks,
)

_METADATA_STRUCT = dy.Struct(
    {
        "module": dy.String(nullable=False),
        "names": dy.String(nullable=False),
        "dots": dy.String(nullable=False),
    },
    nullable=False,
)


class ChunkStagingRow(dy.Schema):
    """Matches the `_stg` view columns consumed by `upsert_chunks.sql`.

    No `embedding` column: embeddings are always NULL on
    initial insert and set later via `update_embedding(s)`.
    """

    repo_id = dy.Int32(nullable=False)
    id = dy.String(nullable=False)
    blob_sha = dy.String(nullable=False)
    file_path = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in ChunkKind)
    name = dy.String(nullable=False)
    scope = dy.String(nullable=False)
    language = dy.String(nullable=False)
    content = dy.String(nullable=False)
    content_tokens = dy.String(nullable=False)
    name_tokens = dy.String(nullable=False)
    line_start = dy.Int32(nullable=False)
    line_end = dy.Int32(nullable=False)
    metadata = _METADATA_STRUCT
    language_plugin_version = dy.Int32(nullable=False)


class EdgeStagingRow(dy.Schema):
    """Matches the `_stg` view columns consumed by `insert_edges.sql`.

    All rows in a batch share the same `commit_sha` and
    `repo_id`; broadcast happens here rather than in SQL.
    """

    repo_id = dy.Int32(nullable=False)
    source_id = dy.String(nullable=False)
    target_id = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in EdgeKind)
    commit_sha = dy.String(nullable=False)


class SnapshotStagingRow(dy.Schema):
    """Matches the `_stg` view columns consumed by `upsert_snapshots.sql`."""

    repo_id = dy.Int32(nullable=False)
    commit_sha = dy.String(nullable=False)
    file_path = dy.String(nullable=False)
    blob_sha = dy.String(nullable=False)
    detected_language = dy.String(nullable=False)


class EmbeddingStagingRow(dy.Schema):
    """Matches the `_emb_stg` view columns consumed by `update_embeddings.sql`.

    `embedding` is a variable-length `List[Float32]`, matching the
    `chunks.embedding FLOAT[]` column.  Its dimension is a runtime
    property of the configured model; `embedding_dim_is_uniform`
    enforces that every vector in a write batch shares one length.
    """

    id = dy.String(nullable=False)
    # `List`, not `Array`: the model's dimension is a runtime value,
    # but `dy.Array` requires a static shape at class definition.
    embedding = dy.List(dy.Float32(), nullable=False)
    embedding_truncated = dy.Bool(nullable=False)

    @dy.rule()
    def embedding_dim_is_uniform(cls) -> pl.Expr:
        lengths = cls.embedding.col.list.len()
        return lengths == lengths.first()


class RepoRefRow(dy.Schema):
    """Backs the cursor-registered `_repo_refs` join view.

    Not an insert target: search/edge SQL joins against this view
    to scope rows to one or more `(repo_id, commit_sha)` snapshots.
    `repo_id` is `Int32` to match the column on `chunks`/`edges`.
    """

    repo_id = dy.Int32(nullable=False)
    commit_sha = dy.String(nullable=False)


# ── Result-row schemas (DuckDB -> Python reads) ──────────────────────


class _ChunkIdentity(dy.Schema):
    """Identity and content columns present in every chunk frame."""

    id = dy.String(nullable=False)
    blob_sha = dy.String(nullable=False)
    file_path = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in ChunkKind)
    name = dy.String(nullable=False)
    scope = dy.String(nullable=False)
    language = dy.String(nullable=False)
    content = dy.String(nullable=False)
    line_start = dy.Int32(nullable=False)
    line_end = dy.Int32(nullable=False)
    metadata = _METADATA_STRUCT


class _SignalColumns(dy.Schema):
    """Per-candidate retrieval signals, normalised by `fuse_scores`."""

    lexical = dy.Float64(nullable=False)
    semantic = dy.Float64(nullable=False)
    importance = dy.Float64(nullable=False)
    proximity = dy.Float64(nullable=False)


class ChunkResultRow(_ChunkIdentity):
    """Columns projected by every chunk-returning SQL file.

    Adds `repo_id` (so a chunk shared by several repos stays a
    distinct row in cross-repo search) and `has_embedding` —
    the existence check that lets `Chunk.embedding` stay a
    sentinel marker without loading the full 1024-float vector.
    """

    repo_id = dy.Int32(nullable=False)
    has_embedding = dy.Bool(nullable=False)


class ChunkContentRow(dy.Schema):
    """A subset of chunk columns for content-only lookups.

    `get_chunks_frame` returns this shape so callers that
    only need identity + source text skip the full
    `ChunkResultRow` round-trip through `list[Chunk]`.
    """

    file_path = dy.String(primary_key=True)
    scope = dy.String(primary_key=True)
    name = dy.String(primary_key=True)
    line_start = dy.UInt32(primary_key=True)
    language = dy.String()
    content = dy.String()


class ScoredChunkResultRow(ChunkResultRow):
    """Chunk projection plus a `score` float from search_* queries."""

    score = dy.Float64(nullable=False)


class ChangedSymbolRow(ChunkResultRow):
    """Chunk projection plus a `change_kind` label from `diff_symbols.sql`.

    Each `UNION ALL` branch of the query selects one side's columns
    as plain references, so the projection matches `ChunkResultRow`
    exactly; validation here fails if that column list drifts. The
    label is `change_kind` (not `change`) because `change` is a SQL
    keyword and cannot be a column alias under the linter.
    """

    change_kind = dy.Enum(k.value for k in ChangeKind)


class EdgeResultRow(dy.Schema):
    """Columns projected by `get_edges.sql`."""

    source_id = dy.String(nullable=False)
    target_id = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in EdgeKind)


class InboundDegreeResultRow(dy.Schema):
    """Columns projected by `inbound_degree.sql`."""

    chunk_id = dy.String(nullable=False)
    degree = dy.Int64(nullable=False)


class ChunkPathResultRow(dy.Schema):
    """Columns projected by `get_chunk_paths.sql`."""

    id = dy.String(nullable=False)
    file_path = dy.String(nullable=False)


class FusionInputRow(ChunkResultRow, _SignalColumns):
    """Retrieval output: DB chunk columns plus signal scores."""


class FusedRow(_ChunkIdentity, _SignalColumns):
    """Fusion output: identity, signals, scoring, resolved embedding.

    `has_embedding` is resolved to `embedding` (sentinel or
    empty list) and all scoring columns are present.  `repo_id`
    is carried through so results can be attributed to their
    repo in cross-repo search.
    """

    repo_id = dy.Int32(nullable=False)
    embedding = dy.List(dy.Float64())
    score = dy.Float64(nullable=False)
    name_match = dy.Float64(nullable=False)
    kind_boost = dy.Float64(nullable=False)
    file_penalty = dy.Float64(nullable=False)
    fusion = dy.Float64(nullable=False)
    reranker = dy.Float64(nullable=False)


def chunks_frame(chunks: list[TokenisedChunk]) -> dy.DataFrame[ChunkStagingRow]:
    """Build a staging frame of tokenised chunks for `_bulk_insert`.

    Each chunk carries its own `repo_id`, so one batch may span
    repos — `upsert_chunks.sql` reads `repo_id` per row.
    """
    if not chunks:
        return ChunkStagingRow.create_empty()
    return pl.DataFrame(TokenisedChunks.dump_python(chunks, mode="json")).pipe(
        ChunkStagingRow.validate, cast=True
    )


def edges_frame(edges: list[Edge], commit_sha: str, repo_id: int) -> dy.DataFrame[EdgeStagingRow]:
    """Build a staging frame of edges scoped to *commit_sha*."""
    if not edges:
        return EdgeStagingRow.create_empty()
    return (
        pl.DataFrame(Edges.dump_python(edges, mode="json"))
        .with_columns(
            repo_id=pl.lit(repo_id, dtype=pl.Int32),
            commit_sha=pl.lit(commit_sha),
        )
        .pipe(EdgeStagingRow.validate, cast=True)
    )


def snapshots_frame(snapshots: list[Snapshot], repo_id: int) -> dy.DataFrame[SnapshotStagingRow]:
    """Build a staging frame from a list of `Snapshot` models."""
    if not snapshots:
        return SnapshotStagingRow.create_empty()
    return (
        pl.DataFrame(Snapshots.dump_python(snapshots, mode="json"))
        .with_columns(repo_id=pl.lit(repo_id, dtype=pl.Int32))
        .pipe(SnapshotStagingRow.validate, cast=True)
    )


def repo_refs_frame(refs: list[RepoRef]) -> dy.DataFrame[RepoRefRow]:
    """Build the `_repo_refs` join view from a list of `RepoRef`."""
    if not refs:
        return RepoRefRow.create_empty()
    return pl.DataFrame(
        {
            "repo_id": [r.repo_id for r in refs],
            "commit_sha": [r.commit_sha for r in refs],
        }
    ).pipe(RepoRefRow.validate, cast=True)


def embeddings_frame(
    ids: list[str],
    embeddings: list[list[float]],
    truncated: list[bool],
) -> dy.DataFrame[EmbeddingStagingRow]:
    """Build a staging frame for `update_embeddings.sql`.

    Paired lists (one embedding per id) -- callers maintain
    correspondence.  All lists must have equal length.
    """
    if not ids:
        return EmbeddingStagingRow.create_empty()
    return pl.DataFrame(
        {"id": ids, "embedding": embeddings, "embedding_truncated": truncated}
    ).pipe(EmbeddingStagingRow.validate, cast=True)


# ── Row → Chunk mapping ────────────────────────────────────────

# Polars Struct dtype matching ImportMeta's fields.  Derived from
# the model so it stays in sync when fields are added.
_IMPORT_META_DTYPE = pl.Struct(dict.fromkeys(ImportMeta.model_fields, pl.String))


def _decode_metadata(frame: pl.DataFrame) -> pl.DataFrame:
    """Decode the `metadata` TEXT column from DuckDB to a Struct.

    Returns plain `pl.DataFrame`: in-place dtype coercion on an
    intermediate frame, not a data boundary.
    """
    if "metadata" not in frame.columns:
        return frame
    if frame["metadata"].dtype == pl.String:
        return frame.with_columns(
            pl.col("metadata").fill_null("{}").str.json_decode(_IMPORT_META_DTYPE)
        )
    return frame


# Truthy sentinel for 'has embedding in DB but not loaded'.
# Chunk.embedding is list[float]: empty = none, non-empty = exists.
# Loading full 1024-float vectors for every chunk is wasteful.
_EMBEDDING_SENTINEL: list[float] = [0.0]


def frame_to_chunks(frame: dy.DataFrame[ChunkResultRow]) -> list[Chunk]:
    """Convert a validated chunk-result frame to `Chunk` models."""
    prepared = (
        frame.with_columns(
            pl.when(pl.col("has_embedding"))
            .then(pl.lit(_EMBEDDING_SENTINEL))
            .otherwise(pl.lit([]))
            .alias("embedding"),
        )
        .drop("has_embedding")
        .to_dicts()
    )
    return Chunks.validate_python(prepared)


def scored_to_chunks(
    frame: dy.DataFrame[ScoredChunkResultRow],
) -> list[tuple[Chunk, float]]:
    """Pair every chunk in *frame* with its `score` column."""
    scores = frame["score"].to_list()
    chunks = frame_to_chunks(frame.drop("score").pipe(ChunkResultRow.validate, cast=True))
    return list(zip(chunks, scores, strict=True))


def changed_to_symbols(
    frame: dy.DataFrame[ChangedSymbolRow],
) -> list[tuple[Chunk, ChangeKind]]:
    """Pair every chunk in *frame* with its `change_kind` label."""
    changes = [ChangeKind(value) for value in frame["change_kind"].to_list()]
    chunks = frame_to_chunks(frame.drop("change_kind").pipe(ChunkResultRow.validate, cast=True))
    return list(zip(chunks, changes, strict=True))
