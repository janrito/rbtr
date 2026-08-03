"""Result-row schemas and read transforms for the index query path.

Schemas here validate the columns DuckDB projects back to Python
(`*ResultRow`), the cursor-registered join views (`_snapshot_refs`,
`_file_paths`, `_serial_map`), and the fusion/ranking frames.  Transforms
map a validated frame to `Chunk` models.  The write/staging schemas live in
`staging.py`; the two are kept independent (see the import-linter CQRS
contracts).

These are pure functions -- they never touch DuckDB directly.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl

from rbtr.domain.models import (
    ChangeKind,
    Chunk,
    ChunkKind,
    Chunks,
    EdgeKind,
    ImportMeta,
    SnapshotRef,
)


class SnapshotRefRow(dy.Schema):
    """Backs the cursor-registered `_snapshot_refs` join view.

    Not an insert target: search/edge SQL joins against this view
    to scope rows to one or more `(repo_id, snapshot_sha)` snapshots.
    `repo_id` is `Int32` to match the column on `file_snapshots`/`edges`.
    """

    repo_id = dy.Int32(nullable=False)
    snapshot_sha = dy.String(nullable=False)


class FilePathRow(dy.Schema):
    """Backs the cursor-registered `_file_paths` join view.

    Not an insert target: the diff SQL inner-joins against this view
    to scope changed symbols to a caller-supplied set of files.
    """

    file_path = dy.String(nullable=False)


class SerialMapRow(dy.Schema):
    """Backs the cursor-registered `_serial_map` join view.

    Not an insert target: `blob_is_current` left-joins a blob's chunks against
    this view to check every chunk is at its language's current
    extraction serial. Maps language id -> current serial (the full
    registry, plus `""` for plaintext). `extraction_serial` is `Int32` to
    match the column on `chunks`.
    """

    language = dy.String(nullable=False)
    extraction_serial = dy.Int32(nullable=False)


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
    file_language = dy.String(nullable=False)
    content = dy.String(nullable=False)
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


class _SignalColumns(dy.Schema):
    """Per-candidate retrieval signals, normalised by `fuse_scores`."""

    lexical = dy.Float64(nullable=False)
    semantic = dy.Float64(nullable=False)
    importance = dy.Float64(nullable=False)
    proximity = dy.Float64(nullable=False)


class ChunkResultRow(_ChunkIdentity):
    """Columns projected by every chunk-returning SQL file.

    Sources `repo_id` from the `file_snapshots` join (so a chunk
    shared by several repos stays a distinct row in cross-repo search)
    and `has_embedding` — whether the chunk has a stored embedding, a
    boolean existence check so reads need not load the full 1024-float
    vector.
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
    source_path = dy.String(nullable=False)
    target_path = dy.String(nullable=False)


class InboundDegreeResultRow(dy.Schema):
    """Columns projected by `inbound_degree.sql`."""

    chunk_id = dy.String(nullable=False)
    degree = dy.Int64(nullable=False)


class InboundRefResultRow(dy.Schema):
    """Columns projected by `inbound_refs.sql`: a referrer + edge kind."""

    name = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in ChunkKind)
    file_path = dy.String(nullable=False)
    line_start = dy.Int32(nullable=False)
    edge = dy.Enum(k.value for k in EdgeKind)


class ChunkPathResultRow(dy.Schema):
    """Columns projected by `get_chunk_paths.sql`."""

    id = dy.String(nullable=False)
    file_path = dy.String(nullable=False)


class FusionInputRow(ChunkResultRow, _SignalColumns):
    """Retrieval output: DB chunk columns plus signal scores."""


class FusedRow(_ChunkIdentity, _SignalColumns):
    """Fusion output: identity, signals, and scoring.

    `has_embedding` marks whether the chunk has a stored embedding.
    `repo_id` is carried through so results can be attributed to their
    repo in cross-repo search.
    """

    repo_id = dy.Int32(nullable=False)
    has_embedding = dy.Bool(nullable=False)
    score = dy.Float64(nullable=False)
    name_match = dy.Float64(nullable=False)
    kind_boost = dy.Float64(nullable=False)
    file_penalty = dy.Float64(nullable=False)
    fusion = dy.Float64(nullable=False)
    reranker = dy.Float64(nullable=False)


def snapshot_refs_frame(refs: list[SnapshotRef]) -> dy.DataFrame[SnapshotRefRow]:
    """Build the `_snapshot_refs` join view from a list of `SnapshotRef`."""
    if not refs:
        return SnapshotRefRow.create_empty()
    return pl.DataFrame(
        {
            "repo_id": [r.repo_id for r in refs],
            "snapshot_sha": [r.snapshot_sha for r in refs],
        }
    ).pipe(SnapshotRefRow.validate, cast=True)


def file_paths_frame(file_paths: list[str]) -> dy.DataFrame[FilePathRow]:
    """Build the `_file_paths` join view from a list of file paths."""
    if not file_paths:
        return FilePathRow.create_empty()
    return pl.DataFrame({"file_path": file_paths}).pipe(FilePathRow.validate, cast=True)


def serial_map_frame(serials: dict[str, int]) -> dy.DataFrame[SerialMapRow]:
    """Build the `_serial_map` join view from a language -> serial map."""
    if not serials:
        return SerialMapRow.create_empty()
    return pl.DataFrame(
        {
            "language": list(serials.keys()),
            "extraction_serial": list(serials.values()),
        }
    ).pipe(SerialMapRow.validate, cast=True)


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


def frame_to_chunks(frame: dy.DataFrame[ChunkResultRow]) -> list[Chunk]:
    """Convert a validated chunk-result frame to `Chunk` models."""
    return Chunks.validate_python(frame.to_dicts())


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
