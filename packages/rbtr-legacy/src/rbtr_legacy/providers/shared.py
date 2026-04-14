"""Shared helpers for provider implementations.

These functions are used by multiple provider modules to avoid
duplicating SDK-specific effort mapping and pricing lookups.
Kept separate from `__init__` so provider modules don't import
their own package init (which imports them).
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from rbtr_legacy.config import ThinkingEffort


def openai_chat_model_settings(model: Model, effort: ThinkingEffort) -> ModelSettings | None:
    """Build `OpenAIChatModelSettings` for *effort*.

    Shared by all providers whose `build_model` returns an
    `OpenAIChatModel` (Fireworks, OpenRouter, custom endpoints).
    """
    # Deferred: openai SDK is heavy; only import when needed.
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

    if not isinstance(model, OpenAIChatModel):
        return None
    match effort:
        case ThinkingEffort.LOW:
            return OpenAIChatModelSettings(openai_reasoning_effort="low")
        case ThinkingEffort.MEDIUM:
            return OpenAIChatModelSettings(openai_reasoning_effort="medium")
        case ThinkingEffort.HIGH:
            return OpenAIChatModelSettings(openai_reasoning_effort="high")
        case ThinkingEffort.MAX:
            return OpenAIChatModelSettings(openai_reasoning_effort="xhigh")
    return None


def openai_responses_model_settings(model: Model, effort: ThinkingEffort) -> ModelSettings | None:
    """Build `OpenAIResponsesModelSettings` for *effort*.

    Shared by providers whose `build_model` returns an
    `OpenAIResponsesModel` (OpenAI, ChatGPT).
    """
    # Deferred: openai SDK is heavy; only import when needed.
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

    if not isinstance(model, OpenAIResponsesModel):
        return None
    match effort:
        case ThinkingEffort.LOW:
            return OpenAIResponsesModelSettings(openai_reasoning_effort="low")
        case ThinkingEffort.MEDIUM:
            return OpenAIResponsesModelSettings(openai_reasoning_effort="medium")
        case ThinkingEffort.HIGH:
            return OpenAIResponsesModelSettings(openai_reasoning_effort="high")
        case ThinkingEffort.MAX:
            return OpenAIResponsesModelSettings(openai_reasoning_effort="xhigh")
    return None


def genai_prices_context_window(genai_id: str, model_id: str) -> int | None:
    """Look up context window from the `genai-prices` snapshot."""
    try:
        # Deferred: loads pricing snapshot from disk on first call.
        from genai_prices.data_snapshot import get_snapshot

        snapshot = get_snapshot()
        _, model_info = snapshot.find_provider_model(model_id, None, genai_id, None)
        ctx = model_info.context_window
        return ctx if isinstance(ctx, int) and ctx > 0 else None
    except (LookupError, ValueError):
        return None
