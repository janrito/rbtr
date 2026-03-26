"""Claude provider — OAuth2 Authorization Code + PKCE flow.

Uses a localhost callback server for the redirect (automatic flow),
with a manual paste-the-URL fallback when the port is busy.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

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
    parse_callback_url,
    run_oauth_flow,
    token_request,
)

# ── Constants ────────────────────────────────────────────────────────

_CLIENT_ID = deobfuscate("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl")
_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"  # noqa: S105
_SCOPES = " ".join(
    [
        "org:create_api_key",
        "user:profile",
        "user:inference",
        "user:sessions:claude_code",
        "user:mcp_servers",
        "user:file_upload",
    ]
)
_OAUTH_BETA = "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14"
_OAUTH_USER_AGENT = "claude-cli/2.1.76 (external, cli)"
_REDIRECT_PORT = 53692
_REDIRECT_PATH = "/callback"
_REDIRECT_URI = f"http://localhost:{_REDIRECT_PORT}{_REDIRECT_PATH}"
_CALLBACK_TIMEOUT_SECONDS = 300


# ── Login flow ────────────────────────────────────────────────────────


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


def _build_auth_params(verifier: str) -> dict[str, str]:
    """Build the authorize URL parameters for PKCE."""
    challenge = make_challenge(verifier)
    return {
        "code": "true",
        "client_id": _CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _REDIRECT_URI,
        "scope": _SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }


def _exchange_code(code: str, verifier: str) -> OAuthCreds:
    """Exchange an authorization code for credentials."""
    data = token_request(
        _TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "client_id": _CLIENT_ID,
            "code": code,
            "state": verifier,
            "redirect_uri": _REDIRECT_URI,
            "code_verifier": verifier,
        },
        as_json=True,
    )
    return _make_oauth(data)


def authenticate(cancel: threading.Event | None = None) -> OAuthCreds:
    """Run the full OAuth + PKCE flow with localhost callback."""
    verifier = make_verifier()
    params = _build_auth_params(verifier)
    code = run_oauth_flow(
        auth_url=_AUTHORIZE_URL,
        params=params,
        port=_REDIRECT_PORT,
        callback_path=_REDIRECT_PATH,
        expected_state=verifier,
        cancel=cancel,
        timeout=_CALLBACK_TIMEOUT_SECONDS,
    )
    return _exchange_code(code, verifier)


# ── Manual fallback (two-phase) ──────────────────────────────────────


def begin_login() -> tuple[str, PendingLogin]:
    """Phase 1: build authorize URL, open browser."""
    verifier = make_verifier()
    params = _build_auth_params(verifier)
    url = build_login_url(_AUTHORIZE_URL, params)
    return url, PendingLogin(code_verifier=verifier)


def complete_login(raw_input: str, pending: PendingLogin) -> OAuthCreds:
    """Phase 2: exchange the pasted redirect URL for credentials."""
    code, _state = parse_callback_url(raw_input)
    return _exchange_code(code, pending.code_verifier)


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


# Required by Anthropic's OAuth endpoint as a **separate first system
# block** — the API rejects non-Haiku models if this text is missing
# or concatenated into a single block with other instructions.
_OAUTH_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def _is_adaptive_thinking(model_id: str) -> bool:
    """Adaptive-thinking models use `thinking: { type: "adaptive" }` instead of budget-based."""
    return "4-6" in model_id or "4.6" in model_id


def _build_oauth_client(access_token: str) -> AsyncAnthropic:
    """Build an Anthropic client that prepends the OAuth identity block.

    The OAuth endpoint rejects non-Haiku models unless the identity
    is a separate first text block in the `system` array.
    """
    # Deferred: anthropic SDK is heavy; only load when this provider is used.
    from anthropic import AsyncAnthropic

    class _OAuthAnthropic(AsyncAnthropic):
        async def post(self, path: str, *, body: object = None, **kwargs: object) -> object:  # type: ignore[override]  # wrapping SDK method
            if isinstance(body, dict) and "system" in body:
                system = body["system"]
                if isinstance(system, str) and system:
                    body = {
                        **body,
                        "system": [
                            {"type": "text", "text": _OAUTH_IDENTITY},
                            {"type": "text", "text": system},
                        ],
                    }
            return await super().post(path, body=body, **kwargs)  # type: ignore[call-overload]  # signature widened

    return _OAuthAnthropic(
        auth_token=access_token,
        default_headers={
            "anthropic-beta": _OAUTH_BETA,
            "user-agent": _OAUTH_USER_AGENT,
            "x-app": "cli",
        },
    )


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
        """Build an Anthropic model using stored OAuth credentials."""
        # Deferred: anthropic SDK is heavy; only load when this provider is used.
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        oauth = ensure_credentials()
        client = _build_oauth_client(oauth.access_token)
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

    def context_window(self, model_id: str) -> int | None:
        """Look up context window from `genai-prices`."""
        from rbtr.providers.shared import genai_prices_context_window

        return genai_prices_context_window(self.GENAI_ID, model_id)


provider = ClaudeProvider()
