"""Embedding model loading and encoding via llama-cpp-python.

The `Embedder` class owns a `GpuModelSlot` that lazily loads
the Llama model and unloads it after an idle timeout.  Domain
logic (embedding, truncation detection) lives here; lifecycle
management is delegated to the slot.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from llama_cpp import LLAMA_POOLING_TYPE_UNSPECIFIED, Llama

from rbtr.config import config
from rbtr.index._gpu_model import GpuModelSlot, install_llama_log_callback, resolve_gguf_path

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EmbedResult:
    """Per-text embedding result with truncation flag."""

    vector: list[float]
    truncated: bool


# ── Model loading ────────────────────────────────────────────────────


def _load_model() -> Llama:
    """Load the embedding model, downloading if necessary."""
    model_path = resolve_gguf_path(config.embedding_model)
    log.info("loading_embedding_model", path=model_path)

    # Install callback before Llama() — ggml_metal_init runs during
    # construction and would otherwise dump to stdout/stderr.
    install_llama_log_callback()

    pooling = config.embedding_pooling_type
    if pooling < 0:
        pooling = LLAMA_POOLING_TYPE_UNSPECIFIED

    model = Llama(
        model_path=str(model_path),
        embedding=True,
        n_ctx=config.embedding_n_ctx,
        n_batch=config.embedding_n_ctx,
        n_gpu_layers=config.embedding_n_gpu_layers,
        verbose=config.embedding_verbose,
        pooling_type=pooling,
    )

    return model


# ── Embedder ─────────────────────────────────────────────────────────


class Embedder:
    """Owns one embedding model instance via a `GpuModelSlot`.

    Lazily loads the model on first use.  The idle monitor is an
    asyncio task spawned by the daemon.  Call `close()` before
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
            label="Embedding",
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

    # ── Embedding ────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[EmbedResult]:
        """Encode a batch of texts into normalised embedding vectors.

        Returns one `EmbedResult` per text with the vector and a
        flag indicating whether the input was truncated to fit the
        context window.  Not thread-safe: callers sharing an
        instance across threads must serialise externally (the
        daemon uses `_gpu_lock` for this).
        """
        if not texts:
            return []
        model = self._slot.get()
        n_ctx = model.n_ctx()
        truncated = [len(model.tokenize(t.encode())) > n_ctx for t in texts]
        vectors: list[list[float]] = model.embed(texts, normalize=True, truncate=True)
        return [EmbedResult(vector=v, truncated=t) for v, t in zip(vectors, truncated, strict=True)]

    def embed_single(self, text: str) -> list[float]:
        """Encode a single text into an embedding vector."""
        return self.embed([text])[0].vector

    def warmup(self) -> None:
        """Eagerly load the model so the first search doesn't pay the cost."""
        self._slot.warmup()


# ── Embedding text formatter ───────────────────────────────────────


def embedding_text(name: str, content: str) -> str:
    """Build the text sent to the embedding model for a chunk.

    Returns `name + newline + content`.  Richer formats were
    tested (header with kind/path, prepended docstring) and both
    produced net-negative results — the docstring is already in
    the content, so prepending it doubles its token weight and
    shifts cosine similarities unpredictably.
    """
    return f"{name}\n{content}"
