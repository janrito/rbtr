"""Embedding model loading and encoding via llama-cpp-python.

The model is lazily initialised on first use and kept in memory
for the lifetime of the process.  GGUF model files are downloaded
via `huggingface_hub` and stored at `~/.rbtr/models/`.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.config import config

if TYPE_CHECKING:
    from llama_cpp import Llama

_log = logging.getLogger(__name__)


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

_llama_log_cb_ref: object = None  # prevent GC of ctypes callback
_model_ref: Llama | None = None
_last_use_time: float = 0.0  # monotonic seconds at last embed call
_idle_monitor_running: bool = False
_idle_stop_event: threading.Event | None = None


def _install_llama_log_callback() -> None:
    """Route all llama.cpp / ggml log output to Python logging."""
    global _llama_log_cb_ref  # module-level callback ref
    from llama_cpp.llama_cpp import llama_log_callback, llama_log_set  # deferred: native lib

    @llama_log_callback
    def _cb(
        level: int,
        text: bytes | None,
        user_data: ctypes.c_void_p,
    ) -> None:
        msg = text.decode(errors="replace").rstrip() if text else ""
        if msg:
            _log.debug("llama.cpp: %s", msg)

    _llama_log_cb_ref = _cb
    llama_log_set(_cb, ctypes.c_void_p())


# ── Singleton ────────────────────────────────────────────────────────


def _silence_hf_hub() -> None:
    """Suppress noisy warnings and progress bars from huggingface_hub."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _download(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download a file from HuggingFace Hub, returning local path.

    Checks the local cache first. Only hits the network if the
    file hasn't been downloaded yet.
    """
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


def _resolve_model_path() -> Path:
    """Download the GGUF file if needed and return its local path.

    Uses `huggingface_hub` for resumable downloads and caching.
    The model ID from `config.embedding_model` is resolved
    to a `<org>/<repo>` HuggingFace repo and a GGUF filename.

    Supports two formats for `config.embedding_model`:

    - `<org>/<repo>/<filename.gguf>` — explicit file.
    - `<org>/<repo>` — picks the first `.gguf` file in the repo.
    """
    model_id = config.embedding_model
    cache_dir = str(config.home / "models")

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
                f"Invalid embedding_model {model_id!r}: "
                "expected '<org>/<repo>' or '<org>/<repo>/<file.gguf>'"
            )
            raise ValueError(msg)

    _log.info("Resolving embedding model %s/%s", repo, filename)
    path = _download(repo, filename, cache_dir)
    return Path(path)


def _default_gguf_filename(repo: str) -> str:
    """Pick the first .gguf file in a HuggingFace repo."""
    for name in _list_repo(repo):
        if name.endswith(".gguf"):
            return name
    msg = f"No .gguf files found in {repo}"
    raise FileNotFoundError(msg)


def _load_model() -> Llama:
    """Load the embedding model, downloading if necessary."""
    model_path = _resolve_model_path()
    _log.info("Loading embedding model from %s", model_path)

    # Install callback before Llama() — ggml_metal_init runs during
    # construction and would otherwise dump to stdout/stderr.
    _install_llama_log_callback()

    from llama_cpp import Llama  # deferred: ~50 MB native lib

    return Llama(
        model_path=str(model_path),
        embedding=True,
        n_ctx=0,  # 0 = use model's trained context length
        n_gpu_layers=-1,  # offload all layers to GPU (Metal on macOS)
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def _get_model_cached() -> Llama:
    """Return the cached model instance, loading on first call."""
    return _load_model()


def get_model() -> Llama:
    """Return the model instance, loading on first call."""
    global _model_ref, _last_use_time
    model = _get_model_cached()
    _model_ref = model
    _last_use_time = time.monotonic()
    return model


def _close_model(model: Llama) -> None:
    """Close a loaded llama model, suppressing teardown-time errors."""
    close_fn = getattr(model, "close", None)
    if not callable(close_fn):
        return
    with contextlib.suppress(Exception):
        close_fn()


def reset_model() -> None:
    """Release the loaded model (useful for tests and cleanup)."""
    global _model_ref, _last_use_time
    if _model_ref is not None:
        _close_model(_model_ref)
        _model_ref = None
    _last_use_time = 0.0
    _get_model_cached.cache_clear()


# ── Embedding text construction ───────────────────────────────────────


def embedding_text(name: str, content: str) -> str:
    """Build the text sent to the embedding model for a chunk.

    Returns `name + newline + content`.  Richer formats were
    tested (header with kind/path, prepended docstring) and both
    produced net-negative results — the docstring is already in
    the content, so prepending it doubles its token weight and
    shifts cosine similarities unpredictably.
    """
    return f"{name}\n{content}"


# ── Encoding ─────────────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode a batch of texts into embedding vectors.

    Returns one vector per input text, normalised for cosine
    similarity.  Empty input returns an empty list.

    Each text is embedded individually to avoid exceeding the
    model's context window — llama-cpp's `embed()` concatenates
    all inputs into a single context, which overflows for batches
    of long code chunks.
    """
    if not texts:
        return []
    model = get_model()
    vectors: list[list[float]] = []
    for text in texts:
        vec = model.embed(text, normalize=True)
        vectors.append(vec)  # type: ignore[arg-type]  # single input returns list[float]
    return vectors


# ── Idle unload ──────────────────────────────────────────────────────


def start_idle_monitor(timeout: float) -> None:
    """Start the idle-unload background thread.

    After *timeout* seconds of inactivity (no embedding calls),
    the loaded model is released.  Set *timeout* to 0 to disable.
    A model that has never been loaded is not loaded by this thread.
    """
    global _idle_monitor_running, _idle_stop_event
    if timeout <= 0:
        return
    if _idle_monitor_running:
        return
    _idle_stop_event = threading.Event()
    _idle_monitor_running = True
    thread = threading.Thread(target=_idle_loop, args=(timeout,), daemon=True)
    thread.start()


def stop_idle_monitor() -> None:
    """Stop the idle-unload background thread."""
    global _idle_monitor_running, _idle_stop_event
    if not _idle_monitor_running or _idle_stop_event is None:
        return
    _idle_stop_event.set()
    _idle_monitor_running = False
    _idle_stop_event = None


def _idle_loop(timeout: float) -> None:
    """Background loop: unload model after *timeout* seconds idle."""
    global _last_use_time, _idle_stop_event
    check_interval = min(timeout, 30.0)  # check at most every 30 s
    while True:
        if _idle_stop_event is not None and _idle_stop_event.wait(timeout=check_interval):
            break
        if _model_ref is None:
            continue  # never loaded, nothing to unload
        idle = time.monotonic() - _last_use_time
        if idle >= timeout:
            _log.info(
                "Embedding model idle for %.0f s (limit: %.0f s), unloading.",
                idle,
                timeout,
            )
            reset_model()


def embed_text(text: str) -> list[float]:
    """Encode a single text into an embedding vector."""
    result = embed_texts([text])
    return result[0]
