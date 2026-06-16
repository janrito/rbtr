"""Shared GPU model lifecycle and GGUF download utilities.

`GpuModelSlot` provides lazy loading, idle-based unloading, and
safe cleanup for any GPU model.  `Embedder` and
future `Reranker` each own a slot via composition — no
subclassing.

The download helpers (`resolve_gguf_path`, etc.) handle
HuggingFace Hub resolution so that each model module only
specifies a config key, not download logic.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import structlog
from llama_cpp.llama_cpp import llama_log_callback, llama_log_set

from rbtr.config import config

# `import logging` is still used below to quiet `huggingface_hub`; this
# module's own logger is structlog (positional-arg calls still work via
# the configured `PositionalArgumentsFormatter`).
log = structlog.get_logger(__name__)


# ── GpuModelSlot ─────────────────────────────────────────────────────


class GpuModelSlot[T]:
    """Lazy-loading, idle-unloading slot for a GPU model.

    Owns one model instance of type *T*, a threading lock for safe
    lazy initialisation, and an asyncio idle-unload task.  The
    external *gpu_lock* (`asyncio.Lock`) serialises all GPU
    inference; the idle monitor acquires it before unloading.

    The slot does not know what *T* is — it loads via *loader*,
    tracks idle time, and calls `close()` on the model if the
    method exists.  Callers get the model via `get()` and own all
    domain-specific logic.

    Args:
        loader:       Callable that creates and returns the model.
        unloader:     Callable that releases model resources.
        idle_timeout: Seconds before the idle monitor unloads the
                      model.  0 disables idle unloading.
        gpu_lock:     Shared asyncio lock for GPU serialisation.
        load_lock:    Shared threading lock for serialising model
                      loads across slots.  When provided, replaces
                      the per-slot ``_model_lock`` so that two
                      slots sharing the same lock cannot trigger
                      concurrent ``ggml_metal_init`` calls.
        label:        Human-readable name for log messages
                      (e.g. `"Embedding"`, `"Reranker"`).
    """

    def __init__(
        self,
        loader: Callable[[], T],
        unloader: Callable[[T], None],
        *,
        idle_timeout: float = 0.0,
        gpu_lock: asyncio.Lock | None = None,
        load_lock: threading.Lock | None = None,
        label: str = "Model",
    ) -> None:
        self._model: T | None = None
        self._model_lock = load_lock or threading.Lock()
        self._loader = loader
        self._unloader = unloader
        self._idle_timeout = idle_timeout
        self._last_use_time: float = 0.0
        self._idle_task: asyncio.Task[None] | None = None
        self._gpu_lock = gpu_lock
        self._label = label

    # ── Public ───────────────────────────────────────────────────

    @property
    def idle_timeout(self) -> float:
        """Configured idle timeout in seconds."""
        return self._idle_timeout

    @property
    def loaded(self) -> bool:
        """Whether the model is currently in memory."""
        return self._model is not None

    def get(self) -> T:
        """Lazily load and return the model.

        Thread-safe via double-checked locking.  Updates the
        last-use timestamp for idle tracking.
        """
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._loader()
        self._last_use_time = time.monotonic()
        return self._model

    def warmup(self) -> None:
        """Eagerly load the model so the first call doesn't pay the cost."""
        self.get()

    def start_idle_monitor(self) -> None:
        """Spawn the asyncio idle-unload task if not already running.

        Must be called from a running event loop.
        """
        if self._idle_timeout <= 0:
            return
        if self._idle_task is not None and not self._idle_task.done():
            return
        self._idle_task = asyncio.create_task(self._idle_loop())

    def stop_idle_monitor(self) -> None:
        """Cancel the idle-unload task."""
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None

    def close(self) -> None:
        """Release the model and stop the idle monitor.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        self.stop_idle_monitor()
        if self._model is not None:
            self._unloader(self._model)
            self._model = None
        self._last_use_time = 0.0

    # ── Internals ────────────────────────────────────────────────

    async def _idle_loop(self) -> None:
        """Asyncio task: unload model after idle timeout."""
        check_interval = min(self._idle_timeout, 30.0)
        while True:
            await asyncio.sleep(check_interval)
            if self._model is None:
                continue
            idle = time.monotonic() - self._last_use_time
            if idle >= self._idle_timeout:
                log.info(
                    "model_idle_unload",
                    label=self._label,
                    idle_seconds=round(idle),
                    limit_seconds=round(self._idle_timeout),
                )
                await self._unload()

    async def _unload(self) -> None:
        """Unload the model, acquiring the GPU lock if set."""
        if self._gpu_lock is not None:
            async with self._gpu_lock:
                self._do_unload()
        else:
            self._do_unload()

    def _do_unload(self) -> None:
        """Release native resources and reset state.

        Re-checks idle time under the lock to avoid a TOCTOU
        race: inference may have used the model between the
        outer check and lock acquisition.
        """
        if self._model is None:
            return
        if time.monotonic() - self._last_use_time < self._idle_timeout:
            return
        model = self._model
        self._model = None
        self._unloader(model)
        self._last_use_time = 0.0


# ── llama.cpp log redirect ───────────────────────────────────────────
#
# The ggml Metal backend writes hundreds of "loaded kernel_..."
# messages plus "embeddings required" warnings to stdout/stderr
# via C-level printf/fprintf.  Redirecting fds is not thread-safe
# (it corrupts Rich's Live display in the main thread).
#
# Instead we use `llama_log_set` to install a callback that routes
# ALL native log output to Python's logging.  The callback must be
# set *before* Llama() construction (which triggers ggml_metal_init).
# We keep a strong reference to prevent GC of the ctypes callback.

_llama_log_cb_ref: ctypes._CData | None = None
"""Strong reference — prevents GC of the ctypes function pointer."""


def install_llama_log_callback() -> None:
    """Route all llama.cpp / ggml log output to Python logging."""
    global _llama_log_cb_ref

    @llama_log_callback
    def _cb(
        level: int,
        text: bytes | None,
        user_data: ctypes.c_void_p,
    ) -> None:
        msg = text.decode(errors="replace").rstrip() if text else ""
        if msg:
            log.debug("llama_cpp", text=msg)

    _llama_log_cb_ref = _cb
    llama_log_set(_cb, ctypes.c_void_p())


# ── GGUF download utilities ─────────────────────────────────────────


def _silence_hf_hub() -> None:
    """Suppress noisy warnings and progress bars from huggingface_hub."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _download(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download a file from HuggingFace Hub, returning its local path."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir)
    if cached is not None:
        return str(cached)
    _silence_hf_hub()
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, token=False)


def _list_repo(repo_id: str) -> list[str]:
    """List files in a HuggingFace repo."""
    from huggingface_hub import list_repo_files  # heavy startup cost

    _silence_hf_hub()
    return list(list_repo_files(repo_id, token=False))


def _default_gguf_filename(repo: str) -> str:
    """Pick the first `.gguf` file in a HuggingFace repo."""
    for name in _list_repo(repo):
        if name.endswith(".gguf"):
            return name
    msg = f"No .gguf files found in {repo}"
    raise FileNotFoundError(msg)


def resolve_gguf_path(model_id: str) -> Path:
    """Resolve a HuggingFace model ID to a local GGUF file path.

    Accepts `"<org>/<repo>/<file.gguf>"` (explicit filename) or
    `"<org>/<repo>"` (picks the first `.gguf` in the repo).
    Downloads the file if not already cached.

    Raises:
        ValueError:     If *model_id* has an unexpected format.
        FileNotFoundError: If no `.gguf` file exists in the repo.
    """
    cache_dir = str(config.cache_dir / "models")

    parts = model_id.split("/")
    match len(parts):
        case 3:
            repo = f"{parts[0]}/{parts[1]}"
            filename = parts[2]
        case 2:
            repo = model_id
            filename = _default_gguf_filename(repo)
        case _:
            msg = (
                f"Invalid model ID {model_id!r}: "
                "expected '<org>/<repo>' or '<org>/<repo>/<file.gguf>'"
            )
            raise ValueError(msg)

    log.info("resolving_model", repo=repo, filename=filename)
    path = _download(repo, filename, cache_dir)
    return Path(path)
