"""Claude provider — OAuth2 Authorization Code + PKCE flow.

Uses Anthropic's hosted callback page — the user authorizes in the
browser, copies the ``code#state`` string, and pastes it back into rbtr.
"""

from __future__ import annotations

import threading
import time
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from urllib.parse import urlencode

import httpx
from pydantic_ai.models import Model

from rbtr.config import config
from rbtr.creds import OAuthCreds, creds
from rbtr.exceptions import RbtrError
from rbtr.oauth import make_challenge, make_verifier, oauth_expired, oauth_is_set

# ── Data ─────────────────────────────────────────────────────────────


@dataclass
class PendingLogin:
    """State kept between the two phases of the copy-paste flow."""

    code_verifier: str


# ── Two-phase login flow ─────────────────────────────────────────────


def _make_oauth(data: dict) -> OAuthCreds:
    """Build ``OAuthCreds`` from a token endpoint response."""
    expires_in = data.get("expires_in")
    return OAuthCreds(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", ""),
        expires_at=time.time() + expires_in if expires_in else None,
    )


def begin_login() -> tuple[str, PendingLogin]:
    """Phase 1: generate PKCE, build the authorize URL, open the browser.

    Returns ``(authorize_url, pending)`` where *pending* must be kept
    until the user pastes the callback code.
    """
    pc = config.providers.claude
    verifier = make_verifier()
    challenge = make_challenge(verifier)

    params = {
        "code": "true",
        "client_id": pc.client_id,
        "response_type": "code",
        "redirect_uri": pc.redirect_uri,
        "scope": pc.scopes,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    authorize_url = f"{pc.authorize_url}?{urlencode(params)}"

    threading.Thread(target=webbrowser.open, args=(authorize_url,), daemon=True).start()

    return authorize_url, PendingLogin(code_verifier=verifier)


def parse_auth_code(raw: str) -> tuple[str, str]:
    """Parse the ``code#state`` string the user pastes.

    Returns ``(code, state)``.
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
    pc = config.providers.claude
    if state != pending.code_verifier:
        raise RbtrError("OAuth state mismatch — possible CSRF attack.")

    resp = httpx.post(
        pc.token_url,
        json={
            "grant_type": "authorization_code",
            "client_id": pc.client_id,
            "code": code,
            "state": state,
            "redirect_uri": pc.redirect_uri,
            "code_verifier": pending.code_verifier,
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if resp.status_code != HTTPStatus.OK:
        raise RbtrError(f"Token exchange failed ({resp.status_code}): {resp.text}")
    return _make_oauth(resp.json())


def _refresh(oauth: OAuthCreds) -> OAuthCreds:
    """Use the refresh token to get a new access token."""
    pc = config.providers.claude
    resp = httpx.post(
        pc.token_url,
        json={
            "grant_type": "refresh_token",
            "client_id": pc.client_id,
            "refresh_token": oauth.refresh_token,
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if resp.status_code != HTTPStatus.OK:
        raise RbtrError(f"Token refresh failed ({resp.status_code}): {resp.text}")
    return _make_oauth(resp.json())


# ── Credential persistence ───────────────────────────────────────────


def ensure_credentials() -> OAuthCreds:
    """Return valid credentials, refreshing if expired.

    Raises ``RbtrError`` if not connected or refresh fails.
    """
    if not oauth_is_set(creds.claude):
        raise RbtrError("Not connected to Anthropic. Use /connect claude.")

    if not oauth_expired(creds.claude):
        return creds.claude

    if not creds.claude.refresh_token:
        raise RbtrError("Access token expired and no refresh token. Use /connect claude.")

    refreshed = _refresh(creds.claude)
    creds.update(claude=refreshed)
    return refreshed


# ── Model listing ────────────────────────────────────────────────────


def list_models() -> list[str]:
    """Fetch available models from the Anthropic API."""
    # Deferred: anthropic SDK is heavy; only load when this provider is used.
    from anthropic import Anthropic, AnthropicError

    pc = config.providers.claude
    oauth = ensure_credentials()
    client = Anthropic(
        auth_token=oauth.access_token,
        default_headers={
            "anthropic-beta": pc.oauth_beta,
            "user-agent": pc.oauth_user_agent,
            "x-app": "cli",
        },
    )
    try:
        page = client.models.list(limit=100)
        return sorted(m.id for m in page.data)
    except (AnthropicError, httpx.HTTPError) as e:
        raise RbtrError(f"Failed to list Claude models ({client.base_url}): {e}") from e


# ── Model construction ───────────────────────────────────────────────


def build_model(model_name: str | None = None) -> Model:
    """Build an Anthropic model using stored OAuth credentials.

    The token is sent as ``Authorization: Bearer <token>`` via the
    Anthropic SDK's ``auth_token`` parameter.  The beta and user-agent
    headers are required for OAuth bearer auth.
    """
    # Deferred: anthropic SDK is heavy; only load when this provider is used.
    from anthropic import AsyncAnthropic
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    pc = config.providers.claude
    oauth = ensure_credentials()
    client = AsyncAnthropic(
        auth_token=oauth.access_token,
        default_headers={
            "anthropic-beta": pc.oauth_beta,
            "user-agent": pc.oauth_user_agent,
            "x-app": "cli",
        },
    )
    provider = AnthropicProvider(anthropic_client=client)
    return AnthropicModel(model_name or pc.default_model, provider=provider)
