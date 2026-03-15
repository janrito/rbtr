"""OpenAI provider — API key auth."""

from __future__ import annotations

import httpx
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from rbtr.config import ThinkingEffort
from rbtr.creds import creds
from rbtr.exceptions import RbtrError


class OpenAIProvider:
    """OpenAI API-key provider."""

    GENAI_ID = "openai"
    LABEL = "OpenAI"
    CRED_FIELD = "openai_api_key"
    KEY_PREFIX = "sk-"

    def is_connected(self) -> bool:
        """Whether an OpenAI API key is stored."""
        return bool(creds.openai_api_key)

    def _require_key(self) -> str:
        if not self.is_connected():
            raise RbtrError("No OpenAI API key stored. Use /connect openai <api_key>.")
        return creds.openai_api_key

    def list_models(self) -> list[str]:
        """Fetch available models from the OpenAI API."""
        # Deferred: openai SDK is heavy; only load when this provider is used.
        from openai import OpenAI, OpenAIError

        key = self._require_key()
        client = OpenAI(api_key=key)
        try:
            page = client.models.list()
            return sorted(m.id for m in page.data)
        except (OpenAIError, httpx.HTTPError) as e:
            raise RbtrError(f"Failed to list OpenAI models ({client.base_url}): {e}") from e

    def build_model(self, model_name: str) -> Model:
        """Build an OpenAI model using the stored API key."""
        # Deferred: openai SDK is heavy; only load when this provider is used.
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider as _OpenAIProvider

        key = self._require_key()
        client = AsyncOpenAI(api_key=key)
        prov = _OpenAIProvider(openai_client=client)
        return OpenAIResponsesModel(model_name, provider=prov)

    def model_settings(
        self, model_id: str, model: Model, effort: ThinkingEffort
    ) -> ModelSettings | None:
        """Build OpenAI Responses thinking-effort settings."""
        from rbtr.providers.shared import openai_responses_model_settings

        return openai_responses_model_settings(model, effort)

    def context_window(self, model_id: str) -> int | None:
        """Look up context window from `genai-prices`."""
        from rbtr.providers.shared import genai_prices_context_window

        return genai_prices_context_window(self.GENAI_ID, model_id)


provider = OpenAIProvider()
