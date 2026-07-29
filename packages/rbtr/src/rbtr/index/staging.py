"""Polars DataFrame builders for the index write path (staging).

Each builder converts a list of domain objects (`Chunk`, `Edge`, or
`Snapshot`) into a typed polars frame whose column names match the
corresponding SQL staging view (`_stg`).  Registering a frame as a virtual
view and running `INSERT INTO ... SELECT` against it is orders of magnitude
faster than `executemany` for large batches; `duckdb.register` accepts polars
frames natively (shared Arrow memory, zero-copy).  Validation via dataframely
catches schema drift at the function boundary.

These are pure functions -- they never touch DuckDB directly.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl

from rbtr.domain.models import (
    ChunkKind,
    Edge,
    EdgeKind,
    Edges,
    Snapshot,
    Snapshots,
    TokenisedChunk,
    TokenisedChunks,
)


class ChunkStagingRow(dy.Schema):
    """Matches the `_stg` view columns consumed by `upsert_chunks.sql`.

    No `repo_id` column: the chunk store is content-addressed and
    shared across repos (repo attribution lives in `file_snapshots`).
    No `embedding` column: embeddings are always NULL on
    initial insert and set later via `update_embedding(s)`.
    """

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
    metadata = dy.Struct(
        {
            "module": dy.String(nullable=False),
            "names": dy.String(nullable=False),
            "dots": dy.String(nullable=False),
        },
        nullable=False,
    )
    extraction_serial = dy.Int32(nullable=False)


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


def chunks_frame(chunks: list[TokenisedChunk]) -> dy.DataFrame[ChunkStagingRow]:
    """Build a staging frame of tokenised chunks for `_bulk_insert`.

    Chunks are content-addressed (keyed by `id`) and shared across
    repos; the batch carries no `repo_id`.
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
