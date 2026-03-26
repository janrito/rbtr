"""Provider registry — dispatches `build_model` by provider prefix.

All provider-specific knowledge (model types, settings classes,
auth checks) is centralised here.  The rest of the engine uses
`build_model`, `model_settings`, and `model_context_window`
without importing any provider module directly.

Provider contract
~~~~~~~~~~~~~~~~~

The `Provider` protocol defines the contract every builtin
provider module must satisfy: `GENAI_ID`, `LABEL`,
`is_connected`, `list_models`, `build_model`,
`model_settings`, and `context_window`.

The `PROVIDERS` dict maps each `BuiltinProvider` to its
module — the **single registration point**.  Adding a new
provider means creating the module, adding a credential field
to `Creds`, and one entry in `PROVIDERS`.

API-key providers are wired through `connect_cmd` so the connect
command and setup auto-detect work without per-provider handlers.
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from rbtr.config import ThinkingEffort
from rbtr.exceptions import RbtrError
from rbtr.providers import claude, endpoint, fireworks, google, openai, openai_codex, openrouter
from rbtr.providers.types import BuiltinProvider, Provider

# ── Provider registry ────────────────────────────────────────────────
#
# Single registration point.  To add a new provider:
#   1. Create `providers/<name>.py` satisfying the `Provider` protocol
#   2. Add a credential field to `Creds`
#   3. Add one entry below
#
# Everything else (connect command, setup auto-detect, model listing,
# genai-prices lookup, completion) derives from this dict.
#
PROVIDERS: dict[BuiltinProvider, Provider] = {
    BuiltinProvider.CLAUDE: claude.provider,
    BuiltinProvider.CHATGPT: openai_codex.provider,
    BuiltinProvider.OPENAI: openai.provider,
    BuiltinProvider.GOOGLE: google.provider,
    BuiltinProvider.FIREWORKS: fireworks.provider,
    BuiltinProvider.OPENROUTER: openrouter.provider,
}


# ── Dispatch ─────────────────────────────────────────────────────────


def build_model(model_name: str) -> Model:
    """Build a model for `<provider>/<model-id>`."""
    prov, model_id = _resolve(model_name)
    return prov.build_model(model_id)


def model_settings(
    model_name: str | None,
    model: Model,
    effort: ThinkingEffort,
) -> ModelSettings | None:
    """Dispatch to the provider's `model_settings`."""
    if not model_name:
        return None
    prov, model_id = _resolve(model_name)
    return prov.model_settings(model_id, model, effort)


def model_context_window(model_name: str | None) -> int | None:
    """Return the context window for *model_name*."""
    if not model_name or "/" not in model_name:
        return None
    try:
        prov, model_id = _resolve(model_name)
    except RbtrError:
        return None
    return prov.context_window(model_id)


# ── Internal helpers ─────────────────────────────────────────────────


def _resolve(model_name: str) -> tuple[Provider, str]:
    """Split `<prefix>/<model-id>` and return `(provider, model_id)`.

    Raises `RbtrError` if the prefix is neither a builtin provider
    nor a stored custom endpoint.
    """
    if "/" not in model_name:
        raise RbtrError(
            f"Invalid model format: {model_name}. "
            "Use <provider>/<model-id>, e.g. claude/claude-sonnet-4-20250514."
        )
    prefix, model_id = model_name.split("/", 1)
    try:
        return PROVIDERS[BuiltinProvider(prefix)], model_id
    except ValueError:
        pass
    ep_prov = endpoint.resolve(prefix)
    if ep_prov is not None:
        return ep_prov, model_id
    raise RbtrError(
        f"Unknown provider: {prefix}. "
        "Use /connect to add a provider, or "
        "/connect endpoint for custom endpoints."
    )
