"""Compatibility shim for llama-cpp-python embedding context bug.

HACK: llama-cpp-python 0.3.23 does not set `kv_unified=True` for
embedding contexts. Without it, `n_seq_max=256` limits each sequence
to `n_ctx / 256` tokens — far too small for code chunks. Batch
embedding of any sequence longer than 256 tokens crashes with
`llama_decode returned 1`.

The fix is `PR #2217`_ (merged to `main`, not yet released). This
module applies the same one-line fix by recreating the context after
construction.

**Remove this entire module** when upgrading to a llama-cpp-python
release that includes #2217. Grep for `_llama_cpp_compat` to find
all call sites.

.. _PR #2217: https://github.com/abetlen/llama-cpp-python/pull/2217
"""

from __future__ import annotations

import llama_cpp._internals as _internals
import structlog
from llama_cpp import Llama

log = structlog.get_logger(__name__)


def fix_embedding_context(model: Llama) -> None:
    """Set `kv_unified=True` on an embedding model's context.

    Must be called immediately after constructing a `Llama` instance
    with `embedding=True`. Recreates the context so that each
    sequence gets the full `n_ctx` window instead of
    `n_ctx / n_seq_max`.

    No-op if `kv_unified` is already `True` (i.e. the upstream
    fix has shipped and this shim is no longer needed).
    """
    if getattr(model.context_params, "kv_unified", True):
        return

    model._ctx.close()  # type: ignore[no-untyped-call]  # llama_cpp C extension, untyped
    model.context_params.kv_unified = True
    model._ctx = _internals.LlamaContext(
        model=model._model,
        params=model.context_params,
        verbose=model.verbose,
    )
    log.debug("applied_kv_unified_fix", n_ctx=model.n_ctx())
