"""Provider registry — dispatches ``build_model`` by provider prefix.

All provider-specific knowledge (model types, settings classes,
auth checks) is centralised here.  The rest of the engine uses
``build_model`` and ``build_model_settings`` without importing
any provider module directly.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from rbtr.config import ThinkingEffort
from rbtr.creds import creds
from rbtr.exceptions import RbtrError
from rbtr.oauth import oauth_is_set
from rbtr.providers import claude, endpoint, openai, openai_codex


class BuiltinProvider(StrEnum):
    """Known first-party provider prefixes."""

    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    OPENAI = "openai"


# ── Model construction ───────────────────────────────────────────────


def build_model(model_name: str | None = None) -> Model:
    """Build a model from whichever provider is connected.

    If *model_name* is given it must be ``<provider>/<model-id>``.
    Without a name, tries providers in priority order.
    """
    if model_name:
        return _build_model_by_name(model_name)

    if oauth_is_set(creds.claude):
        return claude.build_model()

    if oauth_is_set(creds.chatgpt):
        return openai_codex.build_model()

    if creds.openai_api_key:
        return openai.build_model()

    raise RbtrError("No LLM connected. Use /connect claude, chatgpt, or openai.")


def _build_model_by_name(model_name: str) -> Model:
    """Build a model for ``<provider>/<model-id>``."""
    if "/" not in model_name:
        raise RbtrError(
            f"Invalid model format: {model_name}. "
            "Use <provider>/<model-id>, e.g. claude/claude-sonnet-4-20250514."
        )

    prefix, model_id = model_name.split("/", 1)

    try:
        provider = BuiltinProvider(prefix)
    except ValueError:
        # Not a built-in — try custom endpoints
        ep = endpoint.load_endpoint(prefix)
        if ep is None:
            raise RbtrError(
                f"Unknown provider: {prefix}. "
                "Use /connect to add a provider, or "
                "/connect endpoint for custom endpoints."
            ) from None
        return endpoint.build_model(prefix, model_id)

    match provider:
        case BuiltinProvider.CLAUDE:
            if not oauth_is_set(creds.claude):
                raise RbtrError("Not connected to Claude. Use /connect claude.")
            return claude.build_model(model_id)
        case BuiltinProvider.CHATGPT:
            if not oauth_is_set(creds.chatgpt):
                raise RbtrError("Not connected to ChatGPT. Use /connect chatgpt.")
            return openai_codex.build_model(model_id)
        case BuiltinProvider.OPENAI:
            if not creds.openai_api_key:
                raise RbtrError("Not connected to OpenAI. Use /connect openai <api_key>.")
            return openai.build_model(model_id)


# ── Model settings ───────────────────────────────────────────────────


def build_model_settings(
    model: Model,
    effort: ThinkingEffort,
) -> ModelSettings | None:
    """Build provider-specific model settings for *effort*.

    Returns ``None`` when the model doesn't support an effort setting.
    The caller is responsible for recording whether the setting was
    applied (e.g. updating ``session.effort_supported``).
    """
    # Deferred: only import when we need to check the model type.
    from pydantic_ai.models.anthropic import AnthropicModel

    if isinstance(model, AnthropicModel):
        from pydantic_ai.models.anthropic import AnthropicModelSettings

        # Anthropic accepts the same low/medium/high/max literals.
        match effort:
            case ThinkingEffort.LOW:
                return AnthropicModelSettings(anthropic_effort="low")
            case ThinkingEffort.MEDIUM:
                return AnthropicModelSettings(anthropic_effort="medium")
            case ThinkingEffort.HIGH:
                return AnthropicModelSettings(anthropic_effort="high")
            case ThinkingEffort.MAX:
                return AnthropicModelSettings(anthropic_effort="max")

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

    if isinstance(model, OpenAIResponsesModel):
        from pydantic_ai.models.openai import OpenAIResponsesModelSettings

        match effort:
            case ThinkingEffort.LOW:
                return OpenAIResponsesModelSettings(openai_reasoning_effort="low")
            case ThinkingEffort.MEDIUM:
                return OpenAIResponsesModelSettings(openai_reasoning_effort="medium")
            case ThinkingEffort.HIGH:
                return OpenAIResponsesModelSettings(openai_reasoning_effort="high")
            case ThinkingEffort.MAX:
                return OpenAIResponsesModelSettings(openai_reasoning_effort="xhigh")

    if isinstance(model, OpenAIChatModel):
        from pydantic_ai.models.openai import OpenAIChatModelSettings

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
