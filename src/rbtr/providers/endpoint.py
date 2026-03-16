"""Custom OpenAI-compatible endpoint provider.

Models are referenced as `<endpoint-name>/<model-id>`.

Storage functions (`save_endpoint`, `load_endpoint`, etc.) manage
the persisted connection details.  `EndpointProvider` wraps a stored
endpoint as a `Provider` for uniform dispatch.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from rbtr.config import EndpointConfig, ThinkingEffort, config
from rbtr.creds import creds
from rbtr.exceptions import RbtrError

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Endpoint names must be simple identifiers (lowercase, digits, hyphens).
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")

# Max completion tokens when the server's default might be too
# aggressive.  Set to half the context window, capped at 128 k.
_MAX_TOKENS_CAP = 131_072


# ── Types ────────────────────────────────────────────────────────────


@dataclass
class Endpoint:
    """A stored OpenAI-compatible endpoint connection."""

    name: str
    base_url: str
    api_key: str


@dataclass
class ModelMetadata:
    """Metadata for an endpoint model, fetched from `GET /models`."""

    context_window: int


# ── Storage ──────────────────────────────────────────────────────────


def validate_name(name: str) -> None:
    """Raise `RbtrError` if *name* is not a valid endpoint identifier."""
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
    """Load a single endpoint by name.  Returns `None` if not found."""
    ep_cfg = config.endpoints.get(name)
    if ep_cfg is None or not ep_cfg.base_url:
        return None
    return Endpoint(
        name=name,
        base_url=ep_cfg.base_url,
        api_key=creds.endpoint_keys.get(name, ""),
    )


def _require_endpoint(name: str) -> Endpoint:
    """Load an endpoint or raise."""
    ep = load_endpoint(name)
    if ep is None:
        raise RbtrError(f"Endpoint {name!r} not found.")
    return ep


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


# ── Model metadata cache ────────────────────────────────────────────

# Process-lifetime cache: `"endpoint_name/model_id"` → metadata.
_metadata_cache: dict[str, ModelMetadata | None] = {}


def _cache_model_metadata(endpoint_name: str, model: object) -> None:
    """Extract and cache metadata from an OpenAI SDK `Model` object."""
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

    On cache miss, falls back to a `GET /models` call that
    populates the cache for all models the endpoint offers.
    """
    cache_key = f"{endpoint_name}/{model_id}"
    if cache_key in _metadata_cache:
        return _metadata_cache[cache_key]
    try:
        list_models(endpoint_name)
    except RbtrError:
        _metadata_cache[cache_key] = None
        return None
    return _metadata_cache.get(cache_key)


# ── Model listing & construction ─────────────────────────────────────


def list_models(endpoint_name: str) -> list[str]:
    """Fetch available models from a custom endpoint.

    Also populates `_metadata_cache` for every model that
    exposes `context_length`, so `fetch_model_metadata`
    can return instantly afterwards.
    """
    # Deferred: openai SDK is heavy; only load when this provider is used.
    from openai import OpenAI, OpenAIError

    ep = _require_endpoint(endpoint_name)
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


def build_model(endpoint_name: str, model_id: str) -> Model:
    """Build a pydantic-ai `Model` for a custom endpoint.

    Uses the Chat Completions API (not Responses) since most
    OpenAI-compatible endpoints only implement `/chat/completions`.

    The provider subclass overrides `name` so PydanticAI treats
    thinking parts from this endpoint as distinct from OpenAI's own
    Responses API — preventing cross-provider reasoning IDs from
    leaking as extra fields.
    """
    # Deferred: openai SDK is heavy; only load when endpoints are used.
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    ep = _require_endpoint(endpoint_name)

    # Defined inside build_model to keep the openai SDK import deferred.
    class _NamedProvider(OpenAIProvider):
        """`OpenAIProvider` whose `name` is the endpoint name."""

        @property
        def name(self) -> str:
            return endpoint_name

    client = AsyncOpenAI(base_url=ep.base_url, api_key=ep.api_key or "unused")
    prov = _NamedProvider(openai_client=client)
    return OpenAIChatModel(model_id, provider=prov)


# ── Provider protocol implementation ─────────────────────────────────


class EndpointProvider:
    """Wraps a stored endpoint as a `Provider` for uniform dispatch."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.GENAI_ID = name
        self.LABEL = name

    def is_connected(self) -> bool:
        return load_endpoint(self._name) is not None

    def list_models(self) -> list[str]:
        return list_models(self._name)

    def build_model(self, model_id: str) -> Model:
        return build_model(self._name, model_id)

    def model_settings(
        self, model_id: str, model: Model, effort: ThinkingEffort
    ) -> ModelSettings | None:
        """Effort settings + `max_tokens` from endpoint metadata."""
        from rbtr.providers.shared import openai_chat_model_settings

        settings = openai_chat_model_settings(model, effort)
        meta = fetch_model_metadata(self._name, model_id)
        if meta is not None:
            max_tokens = min(meta.context_window // 2, _MAX_TOKENS_CAP)
            settings = {**(settings or {}), "max_tokens": max_tokens}
        return settings

    def system_instructions(self, model_id: str) -> str | None:
        return None

    def context_window(self, model_id: str) -> int | None:
        meta = fetch_model_metadata(self._name, model_id)
        return meta.context_window if meta else None


def resolve(name: str) -> EndpointProvider | None:
    """Return an `EndpointProvider` if *name* is a stored endpoint."""
    if load_endpoint(name) is None:
        return None
    return EndpointProvider(name)
