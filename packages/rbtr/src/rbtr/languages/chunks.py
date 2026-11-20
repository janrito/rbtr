"""Chunking helpers: the span arithmetic every chunker shares, and chunks as rows.

`chunk_rows` and `chunk_models` convert between `Chunk`s and their rows
(`rbtr.domain.rules.ChunkRow`). Actual chunker implementations live in
their language plugin files under `rbtr/languages/`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import dataframely as dy
import polars as pl

from rbtr.domain.models import Chunk, Chunks
from rbtr.domain.rules import ChunkRow, holds_references, no_references

if TYPE_CHECKING:
    from tree_sitter import Node


def last_line(node: Node) -> int:
    """The 1-based number of the last line *node* occupies.

    Tree-sitter points are 0-based, and a node that consumes its
    trailing newline ends at column 0 of the following row, which it
    does not occupy. Adding one to the end row is therefore right for a
    node ending mid-line and one too many for a node ending at a line
    break.

    Every chunker and the query engine share this, so a span means the
    same thing in every language.
    """
    end_row, end_column = node.end_point
    if end_column == 0 and end_row > node.start_point[0]:
        return end_row
    return end_row + 1


# A dumped chunk's `role`, every role's fields present: chunks without an
# import would otherwise infer a struct with no `meta`.
ROLE = pl.Struct({"kind": pl.String, "name": pl.String, "meta": ChunkRow.metadata.dtype})


def chunk_rows(chunks: Sequence[Chunk]) -> dy.DataFrame[ChunkRow]:
    """*chunks* as rows, each role spread over `kind`, `name` and `metadata`.

    A definition or fragment gets empty `refs`: a chunker records no call
    or base.
    """
    if not chunks:
        return ChunkRow.create_empty()
    return (
        pl.DataFrame(Chunks.dump_python(list(chunks), mode="json"), schema_overrides={"role": ROLE})
        .with_columns(
            pl.col("role").struct.field("kind"),
            pl.col("role").struct.field("name").fill_null(""),
            pl.col("role").struct.field("meta").alias("metadata"),
            pl.col("span").struct.field("start").alias("line_start"),
            pl.col("span").struct.field("end").alias("line_end"),
            docs=pl.lit(None, dtype=pl.String),
        )
        .with_columns(
            metadata=pl.when(holds_references())
            .then(
                pl.col("metadata")
                .fill_null(
                    pl.lit(
                        dict.fromkeys(ChunkRow.metadata.inner),
                        dtype=ChunkRow.metadata.dtype,
                    )
                )
                .struct.with_fields(refs=no_references())
            )
            .otherwise("metadata")
        )
        .pipe(ChunkRow.validate, cast=True)
    )


def chunk_models(rows: dy.DataFrame[ChunkRow], file_path: str) -> list[Chunk]:
    """The rows of one file at *file_path* as `Chunk`s, in order."""
    return Chunks.validate_python(
        [
            row | {"file_path": file_path}
            for row in rows.select(
                "blob_sha",
                "file_language",
                "language",
                "scope",
                "content",
                role=pl.struct("kind", "name", meta="metadata"),
                span=pl.struct(start="line_start", end="line_end"),
            ).to_dicts()
        ]
    )
