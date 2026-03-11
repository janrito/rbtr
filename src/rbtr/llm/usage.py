"""LLM usage reading and recording.

Reads cost, token counts, and context-window metadata from
PydanticAI ``ModelResponse`` objects.  Used by the conversation
pipeline (``stream.py``), compaction, and fact extraction.

Distinct from ``rbtr.usage`` which is a pure accumulator with no
LLM / pydantic_ai dependencies.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.usage import RunUsage

from rbtr.providers import model_context_window

from .context import LLMContext


def extract_cost(messages: list[ModelMessage]) -> float:
    """Sum cost from ``ModelResponse.cost()`` across *messages*.

    Skips responses without a ``model_name`` (which can't be priced)
    and silently handles pricing lookup failures.
    """
    total = 0.0
    for msg in messages:
        if not isinstance(msg, ModelResponse) or not msg.model_name:
            continue
        try:
            price = msg.cost()
            total += float(price.total_price)
        except (AssertionError, LookupError, ValueError):
            pass
    return total


def record_run_usage(
    ctx: LLMContext,
    new_messages: list[ModelMessage],
    run_usage: RunUsage,
) -> tuple[float, bool]:
    """Extract cost/context-window from new messages and update session usage.

    Returns ``(run_cost, cost_available)`` so callers can persist cost.

    ``run_usage.input_tokens`` is the **sum** across all requests in
    the run (PydanticAI accumulates).  For context-% we need the
    *last* request's input tokens — that's the actual prompt size the
    model received, reflecting the full conversation + tool results.
    We pull it from the last ``ModelResponse.usage.input_tokens``.
    """
    run_cost = 0.0
    cost_available = False
    context_window: int | None = None
    last_input_tokens: int | None = None

    for msg in new_messages:
        if not isinstance(msg, ModelResponse) or not msg.model_name:
            continue
        # Each ModelResponse carries per-request usage — the last one
        # is the most recent prompt size (what we display as context %).
        last_input_tokens = msg.usage.input_tokens
        try:
            price = msg.cost()
            run_cost += float(price.total_price)
            cost_available = True
            if price.model and price.model.context_window:
                context_window = price.model.context_window
        except (AssertionError, LookupError, ValueError):
            pass

    # Prefer model metadata (endpoint config or genai-prices lookup)
    # over the value from msg.cost(), which may not know the model.
    meta_context_window = model_context_window(ctx.state.model_name)
    if meta_context_window is not None:
        context_window = meta_context_window

    response_count = sum(1 for m in new_messages if isinstance(m, ModelResponse))
    ctx.state.usage.record_run(
        input_tokens=run_usage.input_tokens,
        output_tokens=run_usage.output_tokens,
        cache_read_tokens=run_usage.cache_read_tokens,
        cache_write_tokens=run_usage.cache_write_tokens,
        last_input_tokens=last_input_tokens,
        cost=run_cost,
        cost_available=cost_available,
        context_window=context_window,
        new_responses=response_count,
    )

    return run_cost, cost_available
