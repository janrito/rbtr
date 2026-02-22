"""Custom OpenAI-compatible endpoint provider.

Models are referenced as ``<endpoint-name>/<model-id>``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx
from pydantic_ai.models import Model

from rbtr.config import EndpointConfig, config
from rbtr.creds import creds
from rbtr.exceptions import RbtrError

log = logging.getLogger(__name__)

# Endpoint names must be simple identifiers (lowercase, digits, hyphens).
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")


# ── Types ────────────────────────────────────────────────────────────


@dataclass
class Endpoint:
    """A stored OpenAI-compatible endpoint connection."""

    name: str
    base_url: str
    api_key: str


@dataclass
class ModelMetadata:
    """Metadata for an endpoint model, fetched from ``GET /models``."""

    context_window: int


# ── Persistence ──────────────────────────────────────────────────────


def validate_name(name: str) -> None:
    """Raise ``RbtrError`` if *name* is not a valid endpoint identifier."""
    if not _NAME_RE.match(name):
        raise RbtrError(
            f"Invalid endpoint name: {name!r}. "
            "Use lowercase letters, digits, and hyphens (must start with a letter)."
        )


def save_endpoint(name: str, base_url: str, api_key: str) -> None:
    """Persist an endpoint connection (URL in config, key in creds)."""
    validate_name(name)
    base_url = base_url.rstrip("/")

    config.update(endpoints={**config.endpoints, name: EndpointConfig(base_url=base_url)})

    if api_key:
        creds.update(endpoint_keys={**creds.endpoint_keys, name: api_key})


def load_endpoint(name: str) -> Endpoint | None:
    """Load a single endpoint by name.  Returns None if not found."""
    ep_cfg = config.endpoints.get(name)
    if ep_cfg is None or not ep_cfg.base_url:
        return None
    return Endpoint(
        name=name,
        base_url=ep_cfg.base_url,
        api_key=creds.endpoint_keys.get(name, ""),
    )


def list_endpoints() -> list[Endpoint]:
    """List all stored endpoints, sorted by name."""
    results: list[Endpoint] = []
    for name in sorted(config.endpoints):
        ep_cfg = config.endpoints[name]
        if ep_cfg.base_url:
            results.append(
                Endpoint(
                    name=name,
                    base_url=ep_cfg.base_url,
                    api_key=creds.endpoint_keys.get(name, ""),
                )
            )
    return results


def remove_endpoint(name: str) -> None:
    """Remove a stored endpoint."""
    config.update(endpoints={k: v for k, v in config.endpoints.items() if k != name})
    creds.update(endpoint_keys={k: v for k, v in creds.endpoint_keys.items() if k != name})


# ── Model listing ────────────────────────────────────────────────────


def list_models(endpoint_name: str) -> list[str]:
    """Fetch available models from a custom endpoint.

    Uses the OpenAI ``GET /models`` endpoint.  Also populates
    :data:`_metadata_cache` for every model that exposes
    ``context_length`` in its metadata, so :func:`fetch_model_metadata`
    can return instantly for any model the endpoint offers.

    Returns an empty list if the endpoint doesn't support model listing.
    """
    # Deferred: openai SDK is heavy; only load when this provider is used.
    from openai import OpenAI, OpenAIError

    ep = load_endpoint(endpoint_name)
    if ep is None:
        raise RbtrError(f"Endpoint {endpoint_name!r} not found.")

    client = OpenAI(base_url=ep.base_url, api_key=ep.api_key or "unused")
    try:
        page = client.models.list()
        ids: list[str] = []
        for m in page.data:
            ids.append(m.id)
            _cache_model_metadata(endpoint_name, m)
        return sorted(ids)
    except (OpenAIError, httpx.HTTPError):
        return []


# ── Model metadata ────────────────────────────────────────────────────

# Process-lifetime cache: "endpoint_name/model_id" → metadata.
_metadata_cache: dict[str, ModelMetadata | None] = {}


def _cache_model_metadata(endpoint_name: str, model: object) -> None:
    """Extract and cache metadata from an OpenAI SDK ``Model`` object.

    Called from :func:`list_models` to piggyback on the ``GET /models``
    response that's already being fetched for Tab completion.
    """
    model_id: str = getattr(model, "id", "")
    if not model_id:
        return
    cache_key = f"{endpoint_name}/{model_id}"
    if cache_key in _metadata_cache:
        return
    extra = getattr(model, "model_extra", None) or {}
    metadata = extra.get("metadata") or {}
    ctx = metadata.get("context_length")
    if isinstance(ctx, int) and ctx > 0:
        _metadata_cache[cache_key] = ModelMetadata(context_window=ctx)
    else:
        _metadata_cache[cache_key] = None


def fetch_model_metadata(endpoint_name: str, model_id: str) -> ModelMetadata | None:
    """Return cached metadata for an endpoint model.

    If the cache is empty for this model (e.g. :func:`list_models`
    hasn't run yet), falls back to a ``GET /models`` call and caches
    all results.

    Returns ``None`` when the endpoint doesn't expose metadata or
    the model is not found.
    """
    cache_key = f"{endpoint_name}/{model_id}"
    if cache_key in _metadata_cache:
        return _metadata_cache[cache_key]

    # Cache miss — do a full list (populates cache for all models).
    try:
        list_models(endpoint_name)
    except RbtrError:
        _metadata_cache[cache_key] = None
        return None

    # list_models populated the cache; return whatever it found.
    return _metadata_cache.get(cache_key)


# ── Model construction ───────────────────────────────────────────────


def build_model(endpoint_name: str, model_id: str) -> Model:
    """Build a pydantic-ai Model for a custom endpoint.

    Uses the Chat Completions API (not Responses) since most
    OpenAI-compatible endpoints only implement /chat/completions.
    """
    # Deferred: openai SDK is heavy; only load when this provider is used.
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    ep = load_endpoint(endpoint_name)
    if ep is None:
        raise RbtrError(
            f"Endpoint {endpoint_name!r} not found. "
            "Use /connect endpoint <name> <base_url> <api_key>."
        )

    client = AsyncOpenAI(base_url=ep.base_url, api_key=ep.api_key or "unused")
    provider = OpenAIProvider(openai_client=client)
    return OpenAIModel(model_id, provider=provider)
