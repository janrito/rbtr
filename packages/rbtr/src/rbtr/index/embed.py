"""Index embedding — compute vectors for an already-indexed commit.

`embed_index` resolves the whole work list once, then reads it back a page
at a time and embeds each page in batches.  Each batch commits in its own
write session, so a transaction covers the write alone.

All heavy work runs synchronously in the calling thread — `rbtr watch`
embeds inline, after chunks and edges are committed.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterator

import dataframely as dy
import polars as pl
import structlog

from rbtr.config import config
from rbtr.domain.models import Repo, SnapshotRef
from rbtr.index.embeddings import Embedder, EmbedResult, embedding_text
from rbtr.index.index import Index
from rbtr.index.progress import ProgressCallback, _noop_progress
from rbtr.index.results import ChunkResultRow, EmbedTextRow
from rbtr.index.staging import EmbeddingStagingRow
from rbtr.logging import elapsed_ms

log = structlog.get_logger(__name__)


def embedding_batches(page: dy.DataFrame[ChunkResultRow]) -> Iterator[dy.DataFrame[EmbedTextRow]]:
    """*page*'s chunks in batches of `config.embedding_batch_size`, made one at a time."""
    texts = page.select("id", embedding_text().alias("text"))
    for batch in texts.iter_slices(config.embedding_batch_size):
        yield EmbedTextRow.validate(batch)


def embedded(
    batch: dy.DataFrame[EmbedTextRow], results: list[EmbedResult]
) -> dy.DataFrame[EmbeddingStagingRow]:
    """*batch*'s chunk ids, each with its result's vector and truncation flag.

    *results* are the model's, one per row of *batch* and in its order.
    Raises if their counts differ, or if the vectors differ in length.
    """
    return (
        batch.select("id")
        .with_columns(
            embedding=pl.Series([r.vector for r in results], dtype=pl.List(pl.Float32)),
            embedding_truncated=pl.Series([r.truncated for r in results], dtype=pl.Boolean),
        )
        .pipe(EmbeddingStagingRow.validate)
    )


def write_embedding_batch(
    index: Index, repo: Repo, embeddings: dy.DataFrame[EmbeddingStagingRow]
) -> None:
    """Commit one batch of *embeddings* in its own session.

    The caller supplies writer ownership for repo's project.
    """
    with index.session(repo) as session:
        session.update_embeddings(embeddings)


def embed_index(
    index: Index,
    *,
    at: SnapshotRef,
    embedder: Embedder,
    on_progress: ProgressCallback = _noop_progress,
) -> int:
    """Embed un-embedded chunks for an already-indexed commit.

    Resolves the work list once, then reads it back a page at a time and
    embeds each page in batches.  Each batch commits in its own write
    session, so a transaction covers the write alone.

    Returns the number of chunks that were embedded.
    """
    pending = index.unembedded_chunk_ids(at=at)
    if not pending:
        return 0

    on_progress("loading_model", 0, 0)

    outstanding = len(pending)
    done = 0
    t0 = time.perf_counter()

    for page_ids in itertools.batched(pending, config.embedding_page_size, strict=False):
        page = index.chunks_by_id(list(page_ids), at=at)
        for batch in embedding_batches(page):
            try:
                results = embedder.embed(batch["text"].to_list())
            except (RuntimeError, ValueError):
                log.warning("embedding_batch_failed", exc_info=True)
                continue
            write_embedding_batch(index, at.repo, embedded(batch, results))
            done += len(batch)
            on_progress("embedding", done, outstanding)

    log.info("embedded_chunks", done=done, total=outstanding, elapsed_ms=elapsed_ms(t0))
    return done
