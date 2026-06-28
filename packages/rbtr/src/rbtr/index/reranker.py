"""Cross-encoder reranking via llama-cpp-python.

The `Reranker` class owns a `GpuModelSlot` that lazily loads
a GGUF cross-encoder model and scores `(query, document)` pairs.
After fusion narrows candidates to a pool, the reranker re-scores
each pair with full cross-attention, then blends the result with
the fusion score.

Lifecycle management (lazy load, idle unload) is delegated to
`GpuModelSlot`.  Domain logic (prompt formatting, scoring, blend)
lives here.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

import dataframely as dy
import polars as pl
import structlog
from llama_cpp import LLAMA_POOLING_TYPE_RANK, Llama

from rbtr.config import config
from rbtr.index._gpu_model import GpuModelSlot, install_llama_log_callback, resolve_gguf_path
from rbtr.index.frames import FusedRow
from rbtr.index.search import _normalise_col

log = structlog.get_logger(__name__)

# ── Prompt template ──────────────────────────────────────────────────

_RERANKER_PROMPT = """\
<|im_start|>system
Judge whether the Document meets the requirements based on \
the Query and the Instruct provided. Note that the answer \
can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Given a code search query, retrieve relevant \
code that answers the query
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


# ── Model loading ────────────────────────────────────────────────────


def _load_model() -> Llama:
    """Load the reranker cross-encoder model."""
    model_id = config.reranker_model
    if not model_id:
        msg = "reranker_model is not configured"
        raise ValueError(msg)

    model_path = resolve_gguf_path(model_id)
    log.info("loading_reranker_model", path=model_path)
    install_llama_log_callback()

    model = Llama(
        model_path=str(model_path),
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_RANK,
        n_ctx=config.embedding_n_ctx,
        n_batch=config.embedding_n_ctx,
        n_gpu_layers=config.embedding_n_gpu_layers,
        verbose=config.embedding_verbose,
    )
    return model


# ── Reranker ─────────────────────────────────────────────────────────


class Reranker:
    """Cross-encoder reranker backed by a `GpuModelSlot`.

    Lazily loads the model on first use.  Call `close()` before
    discarding the instance to release native resources cleanly.
    """

    def __init__(
        self,
        idle_timeout: float = 0.0,
        *,
        gpu_lock: asyncio.Lock | None = None,
        load_lock: threading.Lock | None = None,
        model_loader: Callable[[], Llama] | None = None,
    ) -> None:
        self._slot: GpuModelSlot[Llama] = GpuModelSlot(
            model_loader or _load_model,
            lambda model: model.close(),
            idle_timeout=idle_timeout,
            gpu_lock=gpu_lock,
            load_lock=load_lock,
            label="Reranker",
        )

    # ── Lifecycle (delegates to slot) ────────────────────────────

    @property
    def idle_timeout(self) -> float:
        """Configured idle timeout in seconds."""
        return self._slot.idle_timeout

    def start_idle_monitor(self) -> None:
        """Spawn the idle-unload asyncio task if not already running."""
        self._slot.start_idle_monitor()

    def stop_idle_monitor(self) -> None:
        """Cancel the idle-unload task."""
        self._slot.stop_idle_monitor()

    def close(self) -> None:
        """Release the model and stop the idle monitor."""
        self._slot.close()

    def warmup(self) -> None:
        """Eagerly load the model so the first search doesn't pay the cost."""
        self._slot.warmup()

    # ── Reranking ────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        frame: dy.DataFrame[FusedRow],
        top_k: int = 10,
        *,
        blend_weight: float,
    ) -> dy.DataFrame[FusedRow]:
        """Re-score candidates with the cross-encoder.

        Scores each row's `content` against *query*, writes the
        raw scores into the `reranker` column, normalises, blends
        with `fusion`, re-sorts, and trims to *top_k*.

        The `fusion` column is never modified.

        On any model error, returns ``frame.head(top_k)`` in
        fusion order with `reranker` at ``0.0``.
        """
        if frame.is_empty():
            return frame

        w = blend_weight

        try:
            model = self._slot.get()
            t0 = time.perf_counter()
            scores = self._score_candidates(model, query, frame)
            elapsed = time.perf_counter() - t0
            log.info(
                "reranked_candidates",
                candidates=len(scores),
                elapsed_ms=round(elapsed * 1000, 1),
                ms_per_pair=round(elapsed * 1000 / len(scores), 1) if scores else 0,
            )
        except (RuntimeError, ValueError, OSError, TypeError, AttributeError):
            log.warning("reranker_scoring_failed", exc_info=True)
            return FusedRow.validate(frame.head(top_k), cast=True)

        return (
            frame.with_columns(pl.Series("reranker", scores, dtype=pl.Float64))
            .with_columns(_normalise_col("reranker"))
            .with_columns(
                (w * pl.col("fusion") + (1 - w) * pl.col("reranker")).alias("score"),
            )
            .sort("score", "id", descending=[True, False])
            .head(top_k)
            .pipe(FusedRow.validate, cast=True)
        )

    @staticmethod
    def _score_candidates(
        model: Llama,
        query: str,
        frame: dy.DataFrame[FusedRow],
    ) -> list[float]:
        """Score each candidate's content against *query*."""
        contents: list[str] = frame["content"].to_list()
        scores: list[float] = []
        for content in contents:
            prompt = _RERANKER_PROMPT.format(query=query, document=content)
            raw = model.embed(prompt)
            scores.append(float(raw[0]))
        return scores
