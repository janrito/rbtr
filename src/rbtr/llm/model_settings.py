"""Model settings resolution.

Builds merged ``ModelSettings`` from thinking-effort config and
endpoint overrides.  Used by both the main agent run (``llm.py``)
and compaction summaries (``compact.py``).
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from rbtr.config import ThinkingEffort, config
from rbtr.providers import build_model_settings, endpoint_model_settings


def resolve_model_settings(
    model: Model,
    model_name: str | None,
    *,
    effort_supported: bool | None = None,
) -> ModelSettings | None:
    """Build merged model settings (thinking effort + endpoint overrides).

    When *effort_supported* is ``False``, the effort parameter is
    omitted even if ``config.thinking_effort`` is set — this avoids
    re-sending a parameter the model already rejected.
    """
    effort = config.thinking_effort
    if effort is not ThinkingEffort.NONE and effort_supported is not False:
        settings: ModelSettings | None = build_model_settings(model, effort)
    else:
        settings = None

    ep_settings = endpoint_model_settings(model_name)
    if ep_settings is not None:
        settings = {**(settings or {}), **ep_settings}

    return settings
