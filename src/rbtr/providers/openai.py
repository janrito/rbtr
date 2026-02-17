"""OpenAI provider — API key auth."""

from __future__ import annotations

import httpx
from pydantic_ai.models import Model

from rbtr import RbtrError
from rbtr.config import config
from rbtr.creds import creds

# ── Credential persistence ───────────────────────────────────────────


# ── Model listing ────────────────────────────────────────────────────


def list_models() -> list[str]:
    """Fetch available models from the OpenAI API."""
    # Deferred: openai SDK is heavy; only load when this provider is used.
    from openai import OpenAI, OpenAIError

    if not creds.openai_api_key:
        raise RbtrError("No OpenAI API key stored. Use /connect openai <api_key>.")

    client = OpenAI(api_key=creds.openai_api_key)
    try:
        page = client.models.list()
        return sorted(m.id for m in page.data)
    except (OpenAIError, httpx.HTTPError) as e:
        raise RbtrError(f"Failed to list OpenAI models ({client.base_url}): {e}") from e


# ── Model construction ───────────────────────────────────────────────


def build_model(model_name: str | None = None) -> Model:
    """Build an OpenAI model using the stored API key."""
    # Deferred: openai SDK is heavy; only load when this provider is used.
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if not creds.openai_api_key:
        raise RbtrError("No OpenAI API key stored. Use /connect openai <api_key>.")

    client = AsyncOpenAI(api_key=creds.openai_api_key)
    provider = OpenAIProvider(openai_client=client)
    return OpenAIResponsesModel(
        model_name or config.providers.openai.default_model, provider=provider
    )
