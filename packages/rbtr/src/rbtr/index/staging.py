"""The shapes the index write path puts into DuckDB.

Each `staged_*` builder turns what its producer makes into a typed polars
frame whose column names match the corresponding SQL staging view (`_stg`).
Extraction hands over a batch's chunks as `ChunkRow`s, which staging gives
their ids, search tokens and serials; edge inference returns its edges as
an `InferredEdgeRow` frame, which staging only scopes to a snapshot; the
embed loop builds an `EmbeddingStagingRow` frame from each batch; a
snapshot's files arrive as `FileSnapshot`s. Registering a frame as a
virtual view and running `INSERT INTO ... SELECT` against it is orders of
magnitude faster than `executemany` for large batches; `duckdb.register`
accepts polars frames natively (shared Arrow memory, zero-copy).
Validation via dataframely catches schema drift at the function boundary.

These are pure functions -- they never touch DuckDB directly.
"""

from __future__ import annotations

from collections.abc import Mapping

import dataframely as dy
import polars as pl

from rbtr.domain.edge_frames import InferredEdgeRow
from rbtr.domain.identity import chunk_id
from rbtr.domain.models import (
    ChunkKind,
    EdgeKind,
    FileSnapshot,
    FileSnapshots,
    SnapshotRef,
)
from rbtr.domain.rules import ChunkRow
from rbtr.domain.tokenise import search_tokens


class ChunkStagingRow(dy.Schema):
    """Matches the `_stg` view columns consumed by `upsert_chunks.sql`.

    No `repo_id` and no `file_path` column: the chunk store is
    content-addressed and shared across checkouts and paths (both live in
    `file_snapshots`, joined on blob_sha + file_language).
    No `embedding` column: embeddings are always NULL on
    initial insert and set later via `update_embeddings`.
    """

    # One row per id: two in a batch are a merge bug or a hash collision.
    id = dy.String(primary_key=True)
    blob_sha = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in ChunkKind)
    name = dy.String(nullable=False)
    scope = dy.String(nullable=False)
    language = dy.String(nullable=False)
    file_language = dy.String(nullable=False)
    content = dy.String(nullable=False)
    content_tokens = dy.String(nullable=False)
    name_tokens = dy.String(nullable=False)
    start_byte = dy.Int64(nullable=False)
    end_byte = dy.Int64(nullable=False)
    line_start = dy.Int32(nullable=False)
    line_end = dy.Int32(nullable=False)
    # What the role carries beyond kind and name: the union of the role
    # payloads' fields, null where a role has none.
    metadata = ChunkRow.metadata
    extraction_serial = dy.Int32(nullable=False)


class EdgeStagingRow(dy.Schema):
    """Matches the `_stg` view columns consumed by `insert_edges.sql`.

    All rows in a batch share the same `snapshot_sha` and
    `repo_id`; broadcast happens here rather than in SQL.
    """

    repo_id = dy.Int32(nullable=False)
    source_id = dy.String(nullable=False)
    target_id = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in EdgeKind)
    snapshot_sha = dy.String(nullable=False)
    source_path = dy.String(nullable=False)
    target_path = dy.String(nullable=False)


class FileSnapshotStagingRow(dy.Schema):
    """Matches the `_stg` view columns consumed by `upsert_snapshots.sql`."""

    repo_id = dy.Int32(nullable=False)
    snapshot_sha = dy.String(nullable=False)
    file_path = dy.String(nullable=False)
    blob_sha = dy.String(nullable=False)
    detected_language = dy.String(nullable=False)


class EmbeddingStagingRow(dy.Schema):
    """Chunk ids with their embeddings, as `update_embeddings.sql` reads them from `_emb_stg`.

    The embed loop builds it whole from a batch and the model's results,
    so a vector cannot be paired with the wrong chunk.

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


def staged_chunks(
    chunks: dy.DataFrame[ChunkRow], serials: Mapping[str, int]
) -> dy.DataFrame[ChunkStagingRow]:
    """*chunks* as the store's rows: each with its id, its search tokens, and its serial.

    Chunks are content-addressed (keyed by `id`) and shared across repos;
    the batch carries no `repo_id`. A chunk's serial is its own language's
    in *serials* (the embedded language of a block it was read from), else
    its file's, else 1. Ids are computed in one Python pass over their
    columns: polars has no sha256, and its own hash is not stable across
    versions. A blob at two paths of one batch gives its chunks twice, the
    same rows, kept once; two different rows of one id are refused.
    """
    identities = chunks.select("blob_sha", "file_language", "start_byte", "end_byte").iter_rows()
    return (
        chunks.with_columns(
            id=pl.Series([chunk_id(*identity) for identity in identities], dtype=pl.String),
            content_tokens=search_tokens(pl.col("content")),
            name_tokens=search_tokens(pl.col("name")),
            extraction_serial=pl.col("language").replace_strict(
                serials,
                default=pl.col("file_language").replace_strict(
                    serials, default=1, return_dtype=pl.Int32
                ),
                return_dtype=pl.Int32,
            ),
        )
        .select(ChunkStagingRow.column_names())
        .unique(maintain_order=True)
        .pipe(ChunkStagingRow.validate, cast=True)
    )


def staged_edges(
    edges: dy.DataFrame[InferredEdgeRow], *, at: SnapshotRef
) -> dy.DataFrame[EdgeStagingRow]:
    """Build a staging frame of edges scoped to *at*.

    Every row in the batch shares *at*, so `repo_id` and `snapshot_sha`
    are broadcast here rather than repeated per edge in SQL.
    """
    return edges.with_columns(
        repo_id=pl.lit(at.repo.repo_id, dtype=pl.Int32),
        snapshot_sha=pl.lit(at.snapshot_sha),
    ).pipe(EdgeStagingRow.validate, cast=True)


def staged_file_snapshots(
    snapshots: list[FileSnapshot], repo_id: int
) -> dy.DataFrame[FileSnapshotStagingRow]:
    """Build a staging frame from a list of `FileSnapshot` models."""
    if not snapshots:
        return FileSnapshotStagingRow.create_empty()
    return (
        pl.DataFrame(FileSnapshots.dump_python(snapshots, mode="json"))
        .with_columns(repo_id=pl.lit(repo_id, dtype=pl.Int32))
        .pipe(FileSnapshotStagingRow.validate, cast=True)
    )


class SerialMapRow(dy.Schema):
    """Backs the cursor-registered `_serial_map` join view.

    Not an insert target: `current_blobs` left-joins blobs' chunks against
    this view to check every chunk is at its language's current
    extraction serial. Maps language id -> current serial (the full
    registry, plus `""` for plaintext). `extraction_serial` is `Int32` to
    match the column on `chunks`.
    """

    language = dy.String(nullable=False)
    extraction_serial = dy.Int32(nullable=False)


def serial_map_view(serials: dict[str, int]) -> dy.DataFrame[SerialMapRow]:
    """Build the `_serial_map` join view from a language -> serial map."""
    if not serials:
        return SerialMapRow.create_empty()
    return pl.DataFrame(
        {
            "language": list(serials.keys()),
            "extraction_serial": list(serials.values()),
        }
    ).pipe(SerialMapRow.validate, cast=True)
