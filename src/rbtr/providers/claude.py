"""Claude provider — OAuth2 Authorization Code + PKCE flow.

Uses Anthropic's hosted callback page — the user authorizes in the
browser, copies the `code#state` string, and pastes it back into rbtr.
"""

from __future__ import annotations

import time
from typing import Literal

import httpx
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from rbtr.config import ThinkingEffort
from rbtr.creds import OAuthCreds, creds
from rbtr.exceptions import RbtrError
from rbtr.oauth import (
    PendingLogin,
    build_login_url,
    deobfuscate,
    ensure_credentials as _ensure_credentials,
    make_challenge,
    make_verifier,
    oauth_is_set,
    token_request,
)

# ── Constants ────────────────────────────────────────────────────────

_CLIENT_ID = deobfuscate("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl")
_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"  # noqa: S105
_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_SCOPES = "org:create_api_key user:profile user:inference"
_OAUTH_BETA = "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14"
_OAUTH_USER_AGENT = "claude-cli/2.1.62 (external, cli)"


# ── Two-phase login flow ─────────────────────────────────────────────


def _make_oauth(
    data: dict[str, str | int],
    *,
    existing_refresh_token: str = "",
) -> OAuthCreds:
    """Build `OAuthCreds` from a token endpoint response.

    *existing_refresh_token* is preserved when the response doesn't
    include a new refresh token.
    """
    expires_in = data.get("expires_in")
    return OAuthCreds(
        access_token=str(data["access_token"]),
        refresh_token=str(data.get("refresh_token", "")) or existing_refresh_token,
        expires_at=time.time() + int(expires_in) if expires_in else None,
    )


def begin_login() -> tuple[str, PendingLogin]:
    """Phase 1: generate PKCE, build the authorize URL, open the browser.

    Returns `(authorize_url, pending)` where *pending* must be kept
    until the user pastes the callback code.
    """
    verifier = make_verifier()
    challenge = make_challenge(verifier)

    params = {
        "code": "true",
        "client_id": _CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _REDIRECT_URI,
        "scope": _SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    url = build_login_url(_AUTHORIZE_URL, params)
    return url, PendingLogin(code_verifier=verifier)


def parse_auth_code(raw: str) -> tuple[str, str]:
    """Parse the `code#state` string the user pastes.

    Returns `(code, state)`.
    """
    raw = raw.strip()
    if "#" not in raw:
        raise RbtrError(
            "Invalid authorization code format. "
            "Expected code#state — paste the full value from the browser."
        )
    code, state = raw.split("#", 1)
    if not code or not state:
        raise RbtrError(
            "Invalid authorization code format. "
            "Expected code#state — paste the full value from the browser."
        )
    return code, state


def complete_login(code: str, state: str, pending: PendingLogin) -> OAuthCreds:
    """Phase 2: exchange the authorization code for credentials.

    *state* from the callback is validated against the PKCE verifier.
    """
    if state != pending.code_verifier:
        raise RbtrError("OAuth state mismatch — possible CSRF attack.")

    data = token_request(
        _TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "client_id": _CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": _REDIRECT_URI,
            "code_verifier": pending.code_verifier,
        },
        as_json=True,
    )
    return _make_oauth(data)


def _refresh(oauth: OAuthCreds) -> OAuthCreds:
    """Use the refresh token to get a new access token."""
    data = token_request(
        _TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "client_id": _CLIENT_ID,
            "refresh_token": oauth.refresh_token,
        },
        as_json=True,
    )
    return _make_oauth(data, existing_refresh_token=oauth.refresh_token)


# ── Credential persistence ───────────────────────────────────────────


def ensure_credentials() -> OAuthCreds:
    """Return valid credentials, refreshing if expired."""
    return _ensure_credentials("claude", _refresh)


# ── Provider ─────────────────────────────────────────────────────────


# Required by Anthropic's OAuth endpoint — pi and Claude Code both
# send this as the first system block.
_OAUTH_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def _is_adaptive_thinking(model_id: str) -> bool:
    """Adaptive-thinking models use `thinking: { type: "adaptive" }` instead of budget-based."""
    return "4-6" in model_id or "4.6" in model_id


class ClaudeProvider:
    """Claude OAuth provider — satisfies the `Provider` protocol."""

    GENAI_ID = "anthropic"
    LABEL = "Anthropic"

    def is_connected(self) -> bool:
        """Whether Claude OAuth credentials are available."""
        return oauth_is_set(creds.claude)

    def list_models(self) -> list[str]:
        """Fetch available models from the Anthropic API."""
        # Deferred: anthropic SDK is heavy; only load when this provider is used.
        from anthropic import Anthropic, AnthropicError

        oauth = ensure_credentials()
        client = Anthropic(
            auth_token=oauth.access_token,
            default_headers={
                "anthropic-beta": _OAUTH_BETA,
                "user-agent": _OAUTH_USER_AGENT,
                "x-app": "cli",
            },
        )
        try:
            page = client.models.list(limit=100)
            return sorted(m.id for m in page.data)
        except (AnthropicError, httpx.HTTPError) as e:
            raise RbtrError(f"Failed to list Claude models ({client.base_url}): {e}") from e

    def build_model(self, model_name: str) -> Model:
        """Build an Anthropic model using stored OAuth credentials.

        The token is sent as `Authorization: Bearer <token>` via the
        Anthropic SDK's `auth_token` parameter.  The beta and user-agent
        headers are required for OAuth bearer auth.
        """
        # Deferred: anthropic SDK is heavy; only load when this provider is used.
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        oauth = ensure_credentials()
        client = AsyncAnthropic(
            auth_token=oauth.access_token,
            default_headers={
                "anthropic-beta": _OAUTH_BETA,
                "user-agent": _OAUTH_USER_AGENT,
                "x-app": "cli",
            },
        )
        provider = AnthropicProvider(anthropic_client=client)
        return AnthropicModel(model_name, provider=provider)

    def model_settings(
        self, model_id: str, model: Model, effort: ThinkingEffort
    ) -> ModelSettings | None:
        """Build Anthropic thinking-effort settings.

        Adaptive-thinking models (4.6+) require `thinking: { type: "adaptive" }`
        alongside `output_config.effort`. Older models use budget-based thinking.
        """
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

        if not isinstance(model, AnthropicModel):
            return None

        type EffortLevel = Literal["low", "medium", "high", "max"]
        effort_map: dict[ThinkingEffort, EffortLevel] = {
            ThinkingEffort.LOW: "low",
            ThinkingEffort.MEDIUM: "medium",
            ThinkingEffort.HIGH: "high",
            ThinkingEffort.MAX: "max",
        }
        level = effort_map.get(effort)
        if level is None:
            return None

        if _is_adaptive_thinking(model_id):
            return AnthropicModelSettings(
                anthropic_thinking={"type": "adaptive"},
                anthropic_effort=level,
            )

        return AnthropicModelSettings(anthropic_effort=level)

    def system_instructions(self, model_id: str) -> str | None:
        """Anthropic OAuth requires a Claude Code identity prefix."""
        return _OAUTH_IDENTITY

    def context_window(self, model_id: str) -> int | None:
        """Look up context window from `genai-prices`."""
        from rbtr.providers.shared import genai_prices_context_window

        return genai_prices_context_window(self.GENAI_ID, model_id)


provider = ClaudeProvider()
