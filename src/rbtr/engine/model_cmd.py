"""Handler for /model — listing, setting, and caching model IDs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rbtr.config import config
from rbtr.exceptions import RbtrError
from rbtr.providers import (
    PROVIDERS,
    BuiltinProvider,
    endpoint as endpoint_provider,
    model_context_window,
)

if TYPE_CHECKING:
    from .core import Engine

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
    current = engine.state.model_name
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
    if provider not in engine.state.connected_providers:
        engine._warn(f"Unknown provider: {provider}. Use /connect to add a provider first.")
        return

    _refresh_provider(engine, provider)

    # Model list is for display/completion, not gatekeeping.
    # Accept any model — the API will reject invalid ones at call time.
    cached = _cached_models_for(engine, provider)
    if cached and model_id not in cached:
        engine._warn(f"Model {model_id} is not in the known list — setting anyway.")

    _apply_model(engine, model_id)


def _apply_model(engine: Engine, model_id: str) -> None:
    """Persist and activate *model_id*."""
    engine.state.model_name = model_id
    engine.state.effort_supported = None  # re-evaluate on next LLM call
    config.update(model=model_id)

    # Update context window from model metadata so the footer
    # shows the correct value immediately after model switch.
    ctx = model_context_window(model_id)
    if ctx is not None:
        engine.state.usage.context_window = ctx
        engine.state.usage.context_window_known = True

    engine._out(f"Model set to {model_id}")


# ── Cache ────────────────────────────────────────────────────────────


def get_models(engine: Engine, *, force: bool = False) -> list[tuple[str, list[str]]]:
    """Return cached models, refreshing all providers if stale or forced."""
    now = datetime.now(UTC).timestamp()
    if (
        not force
        and engine.state.cached_models
        and now - engine.state.models_fetched_at < _MODEL_CACHE_TTL
    ):
        return engine.state.cached_models

    result: list[tuple[str, list[str]]] = []
    for provider in engine.state.connected_providers:
        models = _fetch_provider_models(engine, provider)
        result.append((provider, models))

    engine.state.cached_models = result
    engine.state.models_fetched_at = now

    # Populate context window from the now-warm metadata caches.
    # Deferred from startup to avoid network calls before the user
    # interacts.  Only runs when the model is set but the context
    # window hasn't been resolved yet.
    if engine.state.model_name and not engine.state.usage.context_window_known:
        ctx = model_context_window(engine.state.model_name)
        if ctx is not None:
            engine.state.usage.context_window = ctx
            engine.state.usage.context_window_known = True

    return result


def _refresh_provider(engine: Engine, provider: str) -> None:
    """Refresh the model list for a single provider.

    Does *not* update ``models_fetched_at`` — this is a partial
    refresh so ``get_models`` will still do a full refresh on the
    next ``/model`` call.
    """
    models = _fetch_provider_models(engine, provider)

    # Replace or append this provider's entry in the cache.
    updated = [(name, ms) for name, ms in engine.state.cached_models if name != provider]
    updated.append((provider, models))
    engine.state.cached_models = updated


def _fetch_provider_models(engine: Engine, provider: str) -> list[str]:
    """Fetch models for a single provider from its API."""
    try:
        prov = PROVIDERS.get(BuiltinProvider(provider)) if provider in BuiltinProvider else None
    except ValueError:
        prov = None
    try:
        if prov is not None:
            return [f"{provider}/{m}" for m in prov.list_models()]
        return [f"{provider}/{m}" for m in endpoint_provider.list_models(provider)]
    except RbtrError as e:
        engine._warn(f"Could not list {provider} models: {e}")
        return []


def _model_in_cache(engine: Engine, model_id: str) -> bool:
    """Check if *model_id* is in the cached model list."""
    return any(model_id in models for _, models in engine.state.cached_models)


def _cached_models_for(engine: Engine, provider: str) -> list[str] | None:
    """Return cached models for *provider*, or None if not cached."""
    for name, models in engine.state.cached_models:
        if name == provider:
            return models
    return None
