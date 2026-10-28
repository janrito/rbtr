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
from collections.abc import Sequence

import structlog

from rbtr.config import config
from rbtr.domain.models import Repo, SnapshotRef
from rbtr.index.embeddings import Embedder, EmbedResult, embedding_text
from rbtr.index.index import Index
from rbtr.index.progress import ProgressCallback, _noop_progress
from rbtr.logging import elapsed_ms

log = structlog.get_logger(__name__)


type EmbeddingBatch = Sequence[tuple[str, EmbedResult]]


def write_embedding_batch(index: Index, repo: Repo, batch: EmbeddingBatch) -> None:
    """Commit chunk IDs paired with their embedding results in one finite session.

    The caller supplies writer ownership for repo's project. Vectors and
    truncation flags stay associated with their chunk IDs through this API.
    """
    with index.session(repo) as session:
        session.update_embeddings(
            [chunk_id for chunk_id, _result in batch],
            [result.vector for _chunk_id, result in batch],
            truncated=[result.truncated for _chunk_id, result in batch],
        )


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
        for batch in itertools.batched(page, config.embedding_batch_size, strict=False):
            texts = [embedding_text(c.name, c.content) for c in batch]
            try:
                results = embedder.embed(texts)
            except (RuntimeError, ValueError):
                log.warning("embedding_batch_failed", exc_info=True)
                continue
            write_embedding_batch(
                index,
                at.repo,
                [(chunk.id, result) for chunk, result in zip(batch, results, strict=True)],
            )
            done += len(batch)
            on_progress("embedding", done, outstanding)

    log.info("embedded_chunks", done=done, total=outstanding, elapsed_ms=elapsed_ms(t0))
    return done
