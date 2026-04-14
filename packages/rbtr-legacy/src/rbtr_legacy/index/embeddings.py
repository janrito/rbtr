"""Embedding model loading and encoding via llama-cpp-python.

The model is lazily initialised on first use and kept in memory
for the lifetime of the process.  GGUF model files are downloaded
via `huggingface_hub` and stored at `config.index.model_cache_dir`
(default `~/.config/rbtr/models/`).
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rbtr_legacy import log
from rbtr_legacy.config import config

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


def _download(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download a file from HuggingFace Hub, returning local path."""
    from huggingface_hub import hf_hub_download  # heavy startup cost

    log.remove_stream_handlers("huggingface_hub")
    # token=False: the embedding model repo is public — explicitly
    # opt out of auth to suppress the "unauthenticated requests" warning.
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, token=False)


def _list_repo(repo_id: str) -> list[str]:
    """List files in a HuggingFace repo."""
    from huggingface_hub import list_repo_files  # heavy startup cost

    log.remove_stream_handlers("huggingface_hub")
    return list(list_repo_files(repo_id))


def _resolve_model_path() -> Path:
    """Download the GGUF file if needed and return its local path.

    Uses `huggingface_hub` for resumable downloads and caching.
    The model ID from `config.index.embedding_model` is resolved
    to a `<org>/<repo>` HuggingFace repo and a GGUF filename.

    Supports two formats for `config.index.embedding_model`:

    - `<org>/<repo>/<filename.gguf>` — explicit file.
    - `<org>/<repo>` — picks the first `.gguf` file in the repo.
    """
    model_id = config.index.embedding_model
    cache_dir = str(Path(config.user_dir) / "models")

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
    global _model_ref
    model = _get_model_cached()
    _model_ref = model
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
    global _model_ref
    if _model_ref is not None:
        _close_model(_model_ref)
        _model_ref = None
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


def embed_text(text: str) -> list[float]:
    """Encode a single text into an embedding vector."""
    result = embed_texts([text])
    return result[0]
