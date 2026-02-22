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

# Max completion tokens when the server's default might be too
# aggressive.  Set to half the context window, capped at 128 k.
_MAX_TOKENS_CAP = 131_072


class BuiltinProvider(StrEnum):
    """Known first-party provider prefixes."""

    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    OPENAI = "openai"

    @property
    def genai_provider_id(self) -> str:
        """Return the ``genai-prices`` provider ID for this provider."""
        match self:
            case BuiltinProvider.CLAUDE:
                return "anthropic"
            case BuiltinProvider.CHATGPT | BuiltinProvider.OPENAI:
                return "openai"


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


def _endpoint_parts(model_name: str | None) -> tuple[str, str] | None:
    """Split *model_name* into ``(endpoint_name, model_id)`` if it
    belongs to a custom endpoint.  Returns ``None`` for built-in
    providers or malformed names.
    """
    if not model_name or "/" not in model_name:
        return None
    prefix, model_id = model_name.split("/", 1)
    try:
        BuiltinProvider(prefix)
        return None  # built-in providers don't use endpoint metadata
    except ValueError:
        return prefix, model_id


def _endpoint_metadata(
    model_name: str | None,
) -> endpoint.ModelMetadata | None:
    """Return cached or freshly fetched metadata for an endpoint model."""
    parts = _endpoint_parts(model_name)
    if parts is None:
        return None
    return endpoint.fetch_model_metadata(*parts)


def endpoint_model_settings(model_name: str | None) -> ModelSettings | None:
    """Build ``ModelSettings`` with a safe ``max_tokens`` for an endpoint model.

    When the endpoint reports a ``context_length`` via ``GET /models``
    metadata, set ``max_tokens`` to half the context window (capped at
    128 k) so the server doesn't pick an aggressive default that blows
    the budget.

    Returns ``None`` when metadata is unavailable or the model is not
    a custom endpoint.
    """
    meta = _endpoint_metadata(model_name)
    if meta is None:
        return None
    max_tokens = min(meta.context_window // 2, _MAX_TOKENS_CAP)
    return ModelSettings(max_tokens=max_tokens)


def endpoint_context_window(model_name: str | None) -> int | None:
    """Return the context window for a custom endpoint model.

    Auto-fetched from the endpoint's ``GET /models`` metadata.
    Returns ``None`` when unavailable.
    """
    meta = _endpoint_metadata(model_name)
    if meta is None:
        return None
    return meta.context_window


def chatgpt_context_window(model_name: str | None) -> int | None:
    """Return the context window for a ChatGPT model.

    Uses metadata from the Codex ``GET /models`` response when
    available. Returns ``None`` when the model is not ChatGPT or
    metadata is unavailable.
    """
    if not model_name or "/" not in model_name:
        return None
    prefix, model_id = model_name.split("/", 1)
    if prefix != BuiltinProvider.CHATGPT:
        return None

    meta = openai_codex.fetch_model_metadata(model_id)
    if meta is None:
        return None
    return meta.context_window


# ── Generic context-window lookup ────────────────────────────────────


def model_context_window(model_name: str | None) -> int | None:
    """Return the context window for *any* model — endpoint or built-in.

    Lookup order:
      1) custom endpoint metadata
      2) ChatGPT Codex model metadata
      3) ``genai-prices`` for built-in providers

    Returns ``None`` when unavailable.
    """
    # Try custom endpoint first.
    ep_ctx = endpoint_context_window(model_name)
    if ep_ctx is not None:
        return ep_ctx

    # ChatGPT can expose context limits from its own /models payload,
    # including models that may not exist in the genai-prices snapshot yet.
    codex_ctx = chatgpt_context_window(model_name)
    if codex_ctx is not None:
        return codex_ctx

    # Try genai-prices for built-in providers.
    if not model_name or "/" not in model_name:
        return None
    prefix, model_id = model_name.split("/", 1)
    try:
        provider = BuiltinProvider(prefix)
    except ValueError:
        return None  # unknown prefix, not a built-in

    try:
        # deferred: loads pricing snapshot from disk on first call
        from genai_prices.data_snapshot import get_snapshot

        snapshot = get_snapshot()
        _, model_info = snapshot.find_provider_model(
            model_id, None, provider.genai_provider_id, None
        )
        ctx = model_info.context_window
        return ctx if isinstance(ctx, int) and ctx > 0 else None
    except (LookupError, ValueError):
        return None
