"""Provider type definitions — Protocol and builtin enum.

Separated from `providers/__init__.py` so modules can import
`BuiltinProvider` or the `Provider` Protocol without loading
all provider submodules or `pydantic_ai`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings

    from rbtr_legacy.config import ThinkingEffort


@runtime_checkable
class Provider(Protocol):
    """Typed contract every builtin provider module must satisfy.

    Each module exposes these as module-level constants and functions.
    The `PROVIDERS` dict stores the modules; attribute access goes
    through the module at call time, so test mocks take effect.
    """

    GENAI_ID: str
    LABEL: str

    def is_connected(self) -> bool: ...
    def list_models(self) -> list[str]: ...
    def build_model(self, model_name: str) -> Model: ...
    def model_settings(
        self, model_id: str, model: Model, effort: ThinkingEffort
    ) -> ModelSettings | None: ...
    def context_window(self, model_id: str) -> int | None: ...


class BuiltinProvider(StrEnum):
    """Known first-party provider prefixes."""

    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    OPENAI = "openai"
    GOOGLE = "google"
    FIREWORKS = "fireworks"
    OPENROUTER = "openrouter"
