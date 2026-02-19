"""Handler for /model — listing, setting, and caching model IDs."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rbtr.config import config
from rbtr.exceptions import RbtrError
from rbtr.providers import (
    BuiltinProvider,
    claude as claude_provider,
    endpoint as endpoint_provider,
    openai as openai_provider,
    openai_codex as codex_provider,
)

if TYPE_CHECKING:
    from .core import Engine

type _ModelLister = Callable[[], list[str]]

_MODEL_CACHE_TTL = 300  # 5 minutes


def cmd_model(engine: Engine, args: str) -> None:
    """List available models or set the active model."""
    model_id = args.strip()
    if not model_id:
        _list_models(engine)
        return
    _set_model(engine, model_id)


# ── Listing ──────────────────────────────────────────────────────────


def _list_models(engine: Engine) -> None:
    """List available models, using cache if fresh."""
    all_models = get_models(engine)
    if not all_models:
        engine._warn("No LLM connected. Use /connect to add a provider first.")
        return
    current = engine.session.model_name
    for provider_name, models in all_models:
        engine._out(f"  {provider_name}:")
        if models:
            for m in models:
                marker = " ◂" if m == current else ""
                engine._out(f"    {m}{marker}")
        else:
            engine._out(f"    /model {provider_name}/<model-id>")


# ── Setting ──────────────────────────────────────────────────────────


def _set_model(engine: Engine, model_id: str) -> None:
    """Switch to *model_id*, refreshing only the necessary provider cache."""
    if "/" not in model_id:
        engine._warn(
            f"Invalid model format: {model_id}. "
            "Use <provider>/<model-id>, e.g. claude/claude-sonnet-4-20250514."
        )
        return

    provider = model_id.split("/", 1)[0]

    # Fast path: model is already in the cache.
    if _model_in_cache(engine, model_id):
        _apply_model(engine, model_id)
        return

    # Identify the provider and refresh only its model list.
    if not _is_known_provider(engine, provider):
        engine._warn(f"Unknown provider: {provider}. Use /connect to add a provider first.")
        return

    _refresh_provider(engine, provider)

    # Validate after refresh. Providers with empty lists accept any ID.
    cached = _cached_models_for(engine, provider)
    if cached is not None and model_id not in cached:
        engine._warn(f"Unknown model: {model_id}")
        engine._out("Available models:")
        for m in cached:
            engine._out(f"    {m}")
        return

    _apply_model(engine, model_id)


def _apply_model(engine: Engine, model_id: str) -> None:
    """Persist and activate *model_id*."""
    engine.session.model_name = model_id
    engine.session.effort_supported = None  # re-evaluate on next LLM call
    config.update(model=model_id)
    engine._out(f"Model set to {model_id}")


# ── Cache ────────────────────────────────────────────────────────────


def get_models(engine: Engine, *, force: bool = False) -> list[tuple[str, list[str]]]:
    """Return cached models, refreshing all providers if stale or forced."""
    now = datetime.now(UTC).timestamp()
    if (
        not force
        and engine.session.cached_models
        and now - engine.session.models_fetched_at < _MODEL_CACHE_TTL
    ):
        return engine.session.cached_models

    result: list[tuple[str, list[str]]] = []
    for provider in _connected_providers(engine):
        models = _fetch_provider_models(engine, provider)
        result.append((provider, models))

    engine.session.cached_models = result
    engine.session.models_fetched_at = now
    return result


def _refresh_provider(engine: Engine, provider: str) -> None:
    """Refresh the model list for a single provider."""
    models = _fetch_provider_models(engine, provider)

    # Replace or append this provider's entry in the cache.
    updated = [(name, ms) for name, ms in engine.session.cached_models if name != provider]
    updated.append((provider, models))
    engine.session.cached_models = updated
    engine.session.models_fetched_at = datetime.now(UTC).timestamp()


def _fetch_provider_models(engine: Engine, provider: str) -> list[str]:
    """Fetch models for a single provider from its API."""
    builtin_listers: dict[str, _ModelLister] = {
        BuiltinProvider.CLAUDE: claude_provider.list_models,
        BuiltinProvider.CHATGPT: codex_provider.list_models,
        BuiltinProvider.OPENAI: openai_provider.list_models,
    }
    try:
        if provider in builtin_listers:
            return [f"{provider}/{m}" for m in builtin_listers[provider]()]
        return [f"{provider}/{m}" for m in endpoint_provider.list_models(provider)]
    except RbtrError as e:
        engine._warn(f"Could not list {provider} models: {e}")
        return []


def _connected_providers(engine: Engine) -> list[str]:
    """Return names of all connected providers."""
    session = engine.session
    providers: list[str] = []
    if session.claude_connected:
        providers.append(BuiltinProvider.CLAUDE)
    if session.chatgpt_connected:
        providers.append(BuiltinProvider.CHATGPT)
    if session.openai_connected:
        providers.append(BuiltinProvider.OPENAI)
    providers.extend(ep.name for ep in endpoint_provider.list_endpoints())
    return providers


def _is_known_provider(engine: Engine, provider: str) -> bool:
    """Check if *provider* is connected or a known endpoint."""
    return provider in _connected_providers(engine)


def _model_in_cache(engine: Engine, model_id: str) -> bool:
    """Check if *model_id* is in the cached model list."""
    return any(model_id in models for _, models in engine.session.cached_models)


def _cached_models_for(engine: Engine, provider: str) -> list[str] | None:
    """Return cached models for *provider*, or None if not cached."""
    for name, models in engine.session.cached_models:
        if name == provider:
            return models
    return None
