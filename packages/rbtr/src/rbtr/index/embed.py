"""Index embedding — compute vectors for an already-indexed commit.

`embed_index` resolves the whole work list once, then reads it back a page
at a time and embeds each page in batches.  Each batch commits in its own
write session, so a transaction covers the write alone.

All heavy work runs synchronously in the calling thread — `rbtr index`
embeds inline, after chunks and edges are committed.
"""

from __future__ import annotations

import itertools
import time

import structlog

from rbtr.config import config
from rbtr.domain.models import SnapshotRef
from rbtr.index.embeddings import Embedder, embedding_text
from rbtr.index.progress import ProgressCallback, _noop_progress
from rbtr.index.store import IndexStore
from rbtr.logging import elapsed_ms

log = structlog.get_logger(__name__)


def embed_index(
    store: IndexStore,
    snapshot_sha: str,
    *,
    repo_id: int,
    embedder: Embedder,
    on_progress: ProgressCallback = _noop_progress,
) -> int:
    """Embed un-embedded chunks for an already-indexed commit.

    Resolves the work list once, then reads it back a page at a time and
    embeds each page in batches.  Each batch commits in its own write
    session, so a transaction covers the write alone.

    Returns the number of chunks that were embedded.
    """
    ref = SnapshotRef(repo_id=repo_id, snapshot_sha=snapshot_sha)
    pending = store.unembedded_chunk_ids(ref)
    if not pending:
        return 0

    on_progress("loading_model", 0, 0)

    outstanding = len(pending)
    done = 0
    t0 = time.perf_counter()

    for page_ids in itertools.batched(pending, config.embedding_page_size, strict=False):
        page = store.get_chunks_by_id(ref, list(page_ids))
        for batch in itertools.batched(page, config.embedding_batch_size, strict=False):
            texts = [embedding_text(c.name, c.content) for c in batch]
            try:
                results = embedder.embed(texts)
            except (RuntimeError, ValueError):
                log.warning("embedding_batch_failed", exc_info=True)
                continue
            with store.session() as session:
                session.update_embeddings(
                    [c.id for c in batch],
                    [r.vector for r in results],
                    truncated=[r.truncated for r in results],
                )
            done += len(batch)
            on_progress("embedding", done, outstanding)

    log.info("embedded_chunks", done=done, total=outstanding, elapsed_ms=elapsed_ms(t0))
    return done
