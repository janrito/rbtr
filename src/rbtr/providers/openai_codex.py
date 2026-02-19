"""ChatGPT provider — OAuth2 Authorization Code + PKCE flow.

Uses a localhost callback server on port 1455.  Falls back to
manual URL paste if the port is busy.
"""

from __future__ import annotations

import base64
import http.server
import json
import secrets
import threading
import time
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from pydantic_ai.models import Model

from rbtr.config import config
from rbtr.creds import OAuthCreds, creds
from rbtr.exceptions import RbtrError
from rbtr.oauth import make_challenge, make_verifier, oauth_expired, oauth_is_set

# ── Data ─────────────────────────────────────────────────────────────


@dataclass
class PendingLogin:
    """State kept between the two phases of the manual fallback flow."""

    code_verifier: str
    state: str


# ── JWT helpers ──────────────────────────────────────────────────────


def _read_account_id(access_token: str) -> str:
    """Extract the ``chatgpt_account_id`` from the JWT payload."""
    pc = config.providers.chatgpt
    try:
        parts = access_token.split(".")
        if len(parts) != 3:
            raise ValueError("Not a JWT")
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        account_id = payload.get(pc.jwt_claim_path, {}).get("chatgpt_account_id")
        if not account_id or not isinstance(account_id, str):
            raise ValueError("No chatgpt_account_id in JWT")
        return account_id
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        raise RbtrError(f"Failed to extract account ID from token: {e}") from e


# ── Localhost callback server ────────────────────────────────────────


class _CallbackServer(http.server.HTTPServer):
    """Receives the OAuth redirect on ``localhost:<port>/auth/callback``."""

    code: str | None
    error: str | None

    def __init__(self, expected_state: str) -> None:
        super().__init__(("127.0.0.1", config.providers.chatgpt.redirect_port), _CallbackHandler)
        self.expected_state = expected_state
        self.code = None
        self.error = None


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    server: _CallbackServer

    def do_GET(self) -> None:
        qs = parse_qs(urlparse(self.path).query)

        if not self.path.startswith("/auth/callback"):
            self._respond(404, "Not found")
            return

        state = qs.get("state", [None])[0]
        if state != self.server.expected_state:
            self.server.error = "State mismatch"
            self._respond(400, "State mismatch")
            return

        error = qs.get("error", [None])[0]
        if error:
            self.server.error = error
            self._respond(400, f"Authentication failed: {error}")
            return

        code = qs.get("code", [None])[0]
        if not code:
            self.server.error = "Missing authorization code"
            self._respond(400, "Missing authorization code")
            return

        self.server.code = code
        self._respond(
            200,
            "Authentication successful. Return to your terminal to continue.",
        )

    def _respond(self, status: int, body: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = (
            "<!DOCTYPE html><html><head><title>rbtr</title></head>"
            f"<body><p>{body}</p></body></html>"
        )
        self.wfile.write(html.encode())

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


# ── Token exchange ───────────────────────────────────────────────────


def _build_auth_params(state: str, verifier: str) -> dict[str, str]:
    """Build the query params for the authorize URL."""
    pc = config.providers.chatgpt
    return {
        "response_type": "code",
        "client_id": pc.client_id,
        "redirect_uri": pc.redirect_uri,
        "scope": pc.scopes,
        "code_challenge": make_challenge(verifier),
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "rbtr",
    }


def _make_oauth(data: dict) -> OAuthCreds:
    """Build ``OAuthCreds`` from a token endpoint response."""
    access_token = data["access_token"]
    if not access_token or not data.get("refresh_token"):
        raise RbtrError("Token response missing required fields.")
    return OAuthCreds(
        access_token=access_token,
        refresh_token=data["refresh_token"],
        expires_at=time.time() + data.get("expires_in", 3600),
        account_id=_read_account_id(access_token),
    )


def _exchange_code(code: str, verifier: str) -> OAuthCreds:
    """Exchange the authorization code for credentials."""
    pc = config.providers.chatgpt
    resp = httpx.post(
        pc.token_url,
        data={
            "grant_type": "authorization_code",
            "client_id": pc.client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": pc.redirect_uri,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code != HTTPStatus.OK:
        raise RbtrError(f"Token exchange failed ({resp.status_code}): {resp.text}")
    return _make_oauth(resp.json())


def _refresh(oauth: OAuthCreds) -> OAuthCreds:
    """Use the refresh token to get a new access token."""
    pc = config.providers.chatgpt
    resp = httpx.post(
        pc.token_url,
        data={
            "grant_type": "refresh_token",
            "client_id": pc.client_id,
            "refresh_token": oauth.refresh_token,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if resp.status_code != HTTPStatus.OK:
        raise RbtrError(f"Token refresh failed ({resp.status_code}): {resp.text}")
    return _make_oauth(resp.json())


# ── Automatic flow (localhost callback) ──────────────────────────────


def authenticate(cancel: threading.Event | None = None) -> OAuthCreds:
    """Run the full OAuth + PKCE flow with localhost callback.

    If the localhost server can't bind (port busy), raises
    ``RbtrError`` with instructions for the manual fallback.
    """
    pc = config.providers.chatgpt
    verifier = make_verifier()
    state = secrets.token_hex(16)

    params = _build_auth_params(state, verifier)
    authorize_url = f"{pc.authorize_url}?{urlencode(params)}"

    try:
        server = _CallbackServer(state)
    except OSError as e:
        raise RbtrError(
            f"Port {pc.redirect_port} is busy ({e}). "
            "Close the conflicting process and try again, or use "
            "/connect openai <api_key> with an API key instead."
        ) from e

    try:
        threading.Thread(target=webbrowser.open, args=(authorize_url,), daemon=True).start()

        server.timeout = 1.0
        deadline = time.time() + pc.callback_timeout_seconds
        while server.code is None and server.error is None:
            if cancel is not None and cancel.is_set():
                raise RbtrError("Cancelled.")
            if time.time() > deadline:
                raise RbtrError("Authorization timed out.")
            server.handle_request()
    finally:
        server.server_close()

    if server.error:
        raise RbtrError(f"Authentication failed: {server.error}")

    if server.code is None:
        raise RbtrError("Authorization completed without a code.")
    return _exchange_code(server.code, verifier)


# ── Manual fallback (two-phase) ──────────────────────────────────────


def begin_login() -> tuple[str, PendingLogin]:
    """Phase 1: build authorize URL, open browser.

    Returns ``(authorize_url, pending)``.
    """
    pc = config.providers.chatgpt
    verifier = make_verifier()
    state = secrets.token_hex(16)

    params = _build_auth_params(state, verifier)
    authorize_url = f"{pc.authorize_url}?{urlencode(params)}"

    threading.Thread(target=webbrowser.open, args=(authorize_url,), daemon=True).start()

    return authorize_url, PendingLogin(code_verifier=verifier, state=state)


def parse_callback_url(raw: str) -> str:
    """Extract the authorization code from user-pasted input.

    Accepts a full redirect URL, a query string, or a bare code.
    """
    raw = raw.strip()
    if not raw:
        raise RbtrError("Empty input. Paste the redirect URL from your browser.")

    # Full URL
    try:
        parsed = urlparse(raw)
        qs = parse_qs(parsed.query)
        code = qs.get("code", [None])[0]
        if code:
            return code
    except ValueError:
        pass

    # Bare code (no URL structure)
    if "=" not in raw and "&" not in raw:
        return raw

    # Query-string format: code=...&state=...
    qs = parse_qs(raw)
    code = qs.get("code", [None])[0]
    if code:
        return code

    raise RbtrError(
        "Could not extract authorization code. Paste the full redirect URL from your browser."
    )


def complete_login(raw_input: str, pending: PendingLogin) -> OAuthCreds:
    """Phase 2: exchange the pasted code for credentials."""
    code = parse_callback_url(raw_input)
    return _exchange_code(code, pending.code_verifier)


# ── Credential persistence ───────────────────────────────────────────


def ensure_credentials() -> OAuthCreds:
    """Return valid credentials, refreshing if expired.

    Raises ``RbtrError`` if not connected or refresh fails.
    """
    if not oauth_is_set(creds.chatgpt):
        raise RbtrError("Not connected to ChatGPT. Use /connect chatgpt.")

    if not oauth_expired(creds.chatgpt):
        return creds.chatgpt

    if not creds.chatgpt.refresh_token:
        raise RbtrError("Access token expired and no refresh token. Use /connect chatgpt.")

    refreshed = _refresh(creds.chatgpt)
    creds.update(chatgpt=refreshed)
    return refreshed


# ── Model listing ────────────────────────────────────────────────────


def list_models() -> list[str]:
    """Fetch available models from the ChatGPT Codex backend.

    The Codex backend returns a non-standard response shape
    (``models[].slug`` instead of ``data[].id``), so we call it
    directly with httpx rather than through the OpenAI SDK.
    """
    pc = config.providers.chatgpt
    oauth = ensure_credentials()
    try:
        r = httpx.get(
            f"{pc.codex_base_url}/models",
            params={"client_version": pc.codex_client_version},
            headers={
                "Authorization": f"Bearer {oauth.access_token}",
                "chatgpt-account-id": oauth.account_id,
                "OpenAI-Beta": "responses=experimental",
                "originator": "rbtr",
            },
            timeout=30.0,
        )
        r.raise_for_status()
        return [m["slug"] for m in r.json()["models"]]
    except (httpx.HTTPError, KeyError) as e:
        raise RbtrError(f"Failed to list ChatGPT models ({pc.codex_base_url}): {e}") from e


# ── Model construction ───────────────────────────────────────────────


def build_model(model_name: str | None = None) -> Model:
    """Build a pydantic-ai Model using the ChatGPT Codex backend.

    The Codex API is OpenAI Responses-compatible at a different URL
    with OAuth bearer auth and account-id headers.
    """
    # Deferred: openai SDK is heavy; only load when this provider is used.
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    pc = config.providers.chatgpt
    oauth = ensure_credentials()
    client = AsyncOpenAI(
        api_key=oauth.access_token,
        base_url=pc.codex_base_url,
        default_headers={
            "chatgpt-account-id": oauth.account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "rbtr",
        },
    )
    provider = OpenAIProvider(openai_client=client)
    return OpenAIResponsesModel(
        model_name or pc.default_model,
        provider=provider,
        settings={"extra_body": {"store": False}},
    )
