"""Index embedding — compute vectors for an already-indexed commit.

`embed_index` fetches un-embedded chunks in pages and processes each page in
batches.  Each batch gets its own write session so the DuckDB write lock is
released between batches — higher-priority builds can run in the gaps.

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

    Fetches unembedded chunks in pages and processes each page
    in batches.  Each batch gets its own write session so the
    DuckDB write lock is released between batches — higher-priority
    builds can run in the gaps.

    Returns the number of chunks that were embedded.
    """
    ref = SnapshotRef(repo_id=repo_id, snapshot_sha=snapshot_sha)
    outstanding = store.chunk_counts_for_snapshot(ref).unembedded
    if outstanding == 0:
        return 0

    on_progress("loading_model", 0, 0)

    done = 0
    t0 = time.perf_counter()

    while missing := store.get_unembedded_chunks(repo_id, snapshot_sha):
        before = done
        for batch in itertools.batched(missing, config.embedding_batch_size, strict=False):
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
        if done == before:
            break

    log.info("embedded_chunks", done=done, total=outstanding, elapsed_ms=elapsed_ms(t0))
    return done
