"""Custom OpenAI-compatible endpoint provider.

Models are referenced as ``<endpoint-name>/<model-id>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
from pydantic_ai.models import Model

from rbtr import RbtrError
from rbtr.config import EndpointConfig, config
from rbtr.creds import creds

# Endpoint names must be simple identifiers (lowercase, digits, hyphens).
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")


# ── Types ────────────────────────────────────────────────────────────


@dataclass
class Endpoint:
    """A stored OpenAI-compatible endpoint connection."""

    name: str
    base_url: str
    api_key: str


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

    Uses the OpenAI ``GET /models`` endpoint. Returns an empty list
    if the endpoint doesn't support model listing.
    """
    # Deferred: openai SDK is heavy; only load when this provider is used.
    from openai import OpenAI, OpenAIError

    ep = load_endpoint(endpoint_name)
    if ep is None:
        raise RbtrError(f"Endpoint {endpoint_name!r} not found.")

    client = OpenAI(base_url=ep.base_url, api_key=ep.api_key or "unused")
    try:
        page = client.models.list()
        return sorted(m.id for m in page.data)
    except (OpenAIError, httpx.HTTPError):
        return []


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
