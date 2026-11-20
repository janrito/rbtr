"""Chunks as rows: `chunk_rows` converts `Chunk`s to their rows (`rbtr.domain.rules.ChunkRow`)."""

from __future__ import annotations

from collections.abc import Sequence

import dataframely as dy
import polars as pl

from rbtr.domain.models import Chunk, Chunks
from rbtr.domain.rules import ChunkRow, holds_references, no_references

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
            pl.col("span").struct.field("start_byte"),
            pl.col("span").struct.field("end_byte"),
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
