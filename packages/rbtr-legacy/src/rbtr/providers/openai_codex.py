"""ChatGPT provider — OAuth2 Authorization Code + PKCE flow.

Uses a localhost callback server on port 1455.  Falls back to
manual URL paste if the port is busy.
"""

from __future__ import annotations

import base64
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from rbtr.config import ThinkingEffort
from rbtr.creds import OAuthCreds, creds
from rbtr.exceptions import RbtrError
from rbtr.oauth import (
    PendingLogin,
    TokenData,
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

_CLIENT_ID = deobfuscate("YXBwX0VNb2FtRUVaNzNmMENrWGFYcDdocmFubg==")
_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
_TOKEN_URL = "https://auth.openai.com/oauth/token"  # noqa: S105
_REDIRECT_URI = "http://localhost:1455/auth/callback"
_REDIRECT_PORT = 1455
_SCOPES = "openid profile email offline_access"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_CLIENT_VERSION = "0.101.0"
_CALLBACK_TIMEOUT_SECONDS = 120
_JWT_CLAIM_PATH = "https://api.openai.com/auth"

# ── Data ─────────────────────────────────────────────────────────────


@dataclass
class ModelMetadata:
    """Metadata for a ChatGPT model, fetched from `GET /models`."""

    context_window: int


# Process-lifetime cache: "model-id" → metadata.
_metadata_cache: dict[str, ModelMetadata | None] = {}


# ── JWT helpers ──────────────────────────────────────────────────────


def _read_account_id(access_token: str) -> str:
    """Extract the `chatgpt_account_id` from the JWT payload."""
    try:
        parts = access_token.split(".")
        if len(parts) != 3:
            raise ValueError("Not a JWT")
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        account_id = payload.get(_JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
        if not account_id or not isinstance(account_id, str):
            raise ValueError("No chatgpt_account_id in JWT")
        return account_id
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        raise RbtrError(f"Failed to extract account ID from token: {e}") from e


# ── Token exchange ───────────────────────────────────────────────────


def _build_auth_params(state: str, verifier: str) -> dict[str, str]:
    """Build the query params for the authorize URL."""
    return {
        "response_type": "code",
        "client_id": _CLIENT_ID,
        "redirect_uri": _REDIRECT_URI,
        "scope": _SCOPES,
        "code_challenge": make_challenge(verifier),
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "rbtr",
    }


def _make_oauth(data: TokenData, *, existing_refresh_token: str = "") -> OAuthCreds:
    """Build `OAuthCreds` from a token endpoint response.

    *existing_refresh_token* is preserved when the response doesn't
    include a new refresh token.
    """
    access_token = str(data["access_token"])
    refresh_token = str(data.get("refresh_token", "")) or existing_refresh_token
    if not access_token or not refresh_token:
        raise RbtrError("Token response missing required fields.")
    return OAuthCreds(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + int(data.get("expires_in", 3600)),
        account_id=_read_account_id(access_token),
    )


def _exchange_code(code: str, verifier: str) -> OAuthCreds:
    """Exchange the authorization code for credentials."""
    data = token_request(
        _TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "client_id": _CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": _REDIRECT_URI,
        },
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
    )
    return _make_oauth(data, existing_refresh_token=oauth.refresh_token)


# ── Automatic flow (localhost callback) ──────────────────────────────


def authenticate(cancel: threading.Event | None = None) -> OAuthCreds:
    """Run the full OAuth + PKCE flow with localhost callback."""
    verifier = make_verifier()
    state = secrets.token_hex(16)
    params = _build_auth_params(state, verifier)
    code = run_oauth_flow(
        auth_url=_AUTHORIZE_URL,
        params=params,
        port=_REDIRECT_PORT,
        callback_path="/auth/callback",
        expected_state=state,
        cancel=cancel,
        timeout=_CALLBACK_TIMEOUT_SECONDS,
    )
    return _exchange_code(code, verifier)


# ── Manual fallback (two-phase) ──────────────────────────────────────


def begin_login() -> tuple[str, PendingLogin]:
    """Phase 1: build authorize URL, open browser."""
    verifier = make_verifier()
    state = secrets.token_hex(16)
    params = _build_auth_params(state, verifier)
    url = build_login_url(_AUTHORIZE_URL, params)
    return url, PendingLogin(code_verifier=verifier, state=state)


def complete_login(raw_input: str, pending: PendingLogin) -> OAuthCreds:
    """Phase 2: exchange the pasted code for credentials."""
    code, _state = parse_callback_url(raw_input)
    return _exchange_code(code, pending.code_verifier)


# ── Credential persistence ───────────────────────────────────────────


def ensure_credentials() -> OAuthCreds:
    """Return valid credentials, refreshing if expired."""
    return _ensure_credentials("chatgpt", _refresh)


# ── Model listing / metadata ─────────────────────────────────────────


def _context_window_from_model(model: dict[str, Any]) -> int | None:
    """Extract a positive context-window value from a model payload.

    The Codex backend is not documented publicly and has changed shape
    over time, so we accept multiple keys and nested objects.
    """
    key_names = {
        "context_window",
        "context_length",
        "max_input_tokens",
        "input_token_limit",
        "max_prompt_tokens",
    }

    stack: list[dict[str, Any]] = [model]
    while stack:
        current = stack.pop()
        for key, value in current.items():
            if key in key_names and isinstance(value, int) and value > 0:
                return value
            if isinstance(value, dict):
                stack.append(value)
    return None


def _cache_model_metadata(model: dict[str, Any]) -> None:
    """Extract and cache metadata for one ChatGPT model payload."""
    slug = model.get("slug")
    if not isinstance(slug, str) or not slug:
        return
    if slug in _metadata_cache:
        return

    ctx = _context_window_from_model(model)
    if ctx is None:
        _metadata_cache[slug] = None
    else:
        _metadata_cache[slug] = ModelMetadata(context_window=ctx)


def fetch_model_metadata(model_id: str) -> ModelMetadata | None:
    """Return cached metadata for a ChatGPT model.

    If metadata is not cached yet, performs a model-list fetch once and
    reuses the populated cache for subsequent lookups.
    """
    if model_id in _metadata_cache:
        return _metadata_cache[model_id]

    try:
        provider.list_models()
    except RbtrError:
        _metadata_cache[model_id] = None
        return None

    return _metadata_cache.get(model_id)


# ── Provider ─────────────────────────────────────────────────────────


class ChatGPTProvider:
    """ChatGPT OAuth provider — satisfies the `Provider` protocol."""

    GENAI_ID = "openai"
    LABEL = "ChatGPT"

    def is_connected(self) -> bool:
        """Whether ChatGPT OAuth credentials are available."""
        return oauth_is_set(creds.chatgpt)

    def list_models(self) -> list[str]:
        """Fetch available models from the ChatGPT Codex backend.

        The Codex backend returns a non-standard response shape
        (`models[].slug` instead of `data[].id`), so we call it
        directly with httpx rather than through the OpenAI SDK.

        Also populates `_metadata_cache` for context-window lookups.
        """
        oauth = ensure_credentials()
        try:
            r = httpx.get(
                f"{_CODEX_BASE_URL}/models",
                params={"client_version": _CODEX_CLIENT_VERSION},
                headers={
                    "Authorization": f"Bearer {oauth.access_token}",
                    "chatgpt-account-id": oauth.account_id,
                    "OpenAI-Beta": "responses=experimental",
                    "originator": "rbtr",
                },
                timeout=30.0,
            )
            r.raise_for_status()

            payload = r.json()
            models = payload.get("models")
            if not isinstance(models, list):
                raise KeyError("models")

            ids: list[str] = []
            for model in models:
                if not isinstance(model, dict):
                    continue
                slug = model.get("slug")
                if isinstance(slug, str) and slug:
                    ids.append(slug)
                _cache_model_metadata(model)
            return ids
        except (httpx.HTTPError, KeyError, AttributeError) as e:
            raise RbtrError(f"Failed to list ChatGPT models ({_CODEX_BASE_URL}): {e}") from e

    def build_model(self, model_name: str) -> Model:
        """Build a pydantic-ai Model using the ChatGPT Codex backend.

        The Codex API is OpenAI Responses-compatible at a different URL
        with OAuth bearer auth and account-id headers.
        """
        # Deferred: openai SDK is heavy; only load when this provider is used.
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        oauth = ensure_credentials()
        client = AsyncOpenAI(
            api_key=oauth.access_token,
            base_url=_CODEX_BASE_URL,
            default_headers={
                "chatgpt-account-id": oauth.account_id,
                "OpenAI-Beta": "responses=experimental",
                "originator": "rbtr",
            },
        )
        prov = OpenAIProvider(openai_client=client)
        return OpenAIResponsesModel(
            model_name,
            provider=prov,
            settings={"extra_body": {"store": False}},
        )

    def model_settings(
        self, model_id: str, model: Model, effort: ThinkingEffort
    ) -> ModelSettings | None:
        """Build OpenAI Responses thinking-effort settings."""
        from rbtr.providers.shared import openai_responses_model_settings

        return openai_responses_model_settings(model, effort)

    def context_window(self, model_id: str) -> int | None:
        """Look up context window — Codex metadata first, then `genai-prices`."""
        meta = fetch_model_metadata(model_id)
        if meta is not None:
            return meta.context_window
        from rbtr.providers.shared import genai_prices_context_window

        return genai_prices_context_window(self.GENAI_ID, model_id)


provider = ChatGPTProvider()
