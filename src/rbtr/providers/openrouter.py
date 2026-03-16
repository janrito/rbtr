"""OpenRouter provider — API key auth."""

from __future__ import annotations

import httpx
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from rbtr.config import ThinkingEffort
from rbtr.creds import creds
from rbtr.exceptions import RbtrError

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    """OpenRouter API-key provider."""

    GENAI_ID = "openrouter"
    LABEL = "OpenRouter"
    CRED_FIELD = "openrouter_api_key"

    def is_connected(self) -> bool:
        """Whether an OpenRouter API key is available."""
        return bool(creds.openrouter_api_key)

    def _require_key(self) -> str:
        if not self.is_connected():
            raise RbtrError(
                "No OpenRouter API key. Use /connect openrouter <api_key>"
                " or set OPENROUTER_API_KEY."
            )
        return creds.openrouter_api_key

    def list_models(self) -> list[str]:
        """Fetch available models from the OpenRouter API."""
        from openai import OpenAI, OpenAIError

        key = self._require_key()
        client = OpenAI(base_url=_BASE_URL, api_key=key)
        try:
            page = client.models.list()
            return sorted(m.id for m in page.data)
        except (OpenAIError, httpx.HTTPError) as e:
            raise RbtrError(f"Failed to list OpenRouter models: {e}") from e

    def build_model(self, model_name: str) -> Model:
        """Build a model via `OpenRouterProvider`."""
        # Deferred: openai SDK is heavy; only load when this provider is used.
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider as _OpenRouterProvider

        key = self._require_key()
        prov = _OpenRouterProvider(api_key=key)
        return OpenRouterModel(model_name, provider=prov)

    def model_settings(
        self, model_id: str, model: Model, effort: ThinkingEffort
    ) -> ModelSettings | None:
        """Build OpenRouter thinking-effort settings.

        OpenRouter proxies multiple backends; pass through the Chat
        API `reasoning_effort` and let the backend decide.
        """
        from rbtr.providers.shared import openai_chat_model_settings

        return openai_chat_model_settings(model, effort)

    def system_instructions(self, model_id: str) -> str | None:
        return None

    def context_window(self, model_id: str) -> int | None:
        """Look up context window from `genai-prices`."""
        from rbtr.providers.shared import genai_prices_context_window

        return genai_prices_context_window(self.GENAI_ID, model_id)


provider = OpenRouterProvider()
