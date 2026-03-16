"""Google provider — Gemini CLI OAuth via Cloud Code Assist.

Uses the same PKCE + localhost callback flow as the Gemini CLI:
token exchange, Cloud Code project discovery.  All inference calls
route through the Cloud Code Assist API — the only endpoint enabled
on auto-provisioned projects.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import AsyncIterator

import anyio
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
    parse_callback_url,
    run_oauth_flow,
    token_request,
)

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_CLIENT_ID = deobfuscate(
    "NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVq"
    "LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29t"
)

_CLIENT_SECRET = deobfuscate("R09DU1BYLTR1SGdNUG0tMW83U2stZ2VWNkN1NWNsWEZzeGw=")

_REDIRECT_URI = "http://localhost:8085/oauth2callback"
_REDIRECT_PORT = 8085
_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform "
    "https://www.googleapis.com/auth/userinfo.email "
    "https://www.googleapis.com/auth/userinfo.profile"
)
_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
_CODE_ASSIST_URL = "https://cloudcode-pa.googleapis.com"

# ── Cloud Code Assist transport ──────────────────────────────────────

_CCA_GENERATE_URL = f"{_CODE_ASSIST_URL}/v1internal:streamGenerateContent"


_CCA_HEADERS = {
    "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "X-Goog-Api-Client": "gl-python",
    "Client-Metadata": json.dumps(
        {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    ),
}

_MODEL_RE = re.compile(r"models/([^:/?]+)")


def _extract_model_name(url: str) -> str:
    """Extract model name from a Vertex AI URL.

    E.g. `.../publishers/google/models/gemini-2.5-pro:streamGenerateContent`
    → `gemini-2.5-pro`.
    """
    m = _MODEL_RE.search(url)
    return m.group(1) if m else ""


class _CCAStream(httpx.AsyncByteStream):
    """Transform CCA SSE events by unwrapping the `response` envelope.

    CCA returns `data: {"response": {…}}`; the google-genai SDK
    expects `data: {…}`.
    """

    def __init__(self, inner: httpx.AsyncByteStream) -> None:
        self._inner = inner

    async def __aiter__(self) -> AsyncIterator[bytes]:
        buffer = b""
        async for chunk in self._inner:
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.rstrip(b"\r")
                if line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:])
                        inner = data.get("response", data)
                        yield b"data: " + json.dumps(inner).encode() + b"\n"
                    except (json.JSONDecodeError, TypeError):
                        yield line + b"\n"
                else:
                    yield line + b"\n"
        if buffer:
            yield buffer

    async def aclose(self) -> None:
        await self._inner.aclose()


class _CCATransport(httpx.AsyncBaseTransport):
    """Redirect google-genai SDK requests to Cloud Code Assist.

    The SDK formats the request body for Vertex AI (camelCase, correct
    structure).  This transport wraps the body in the CCA envelope
    `{project, model, request: …}` and rewrites the URL.  The SSE
    response is unwrapped by `_CCAStream`.
    """

    def __init__(self, project_id: str) -> None:
        self._project_id = project_id
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Offload to a thread: ensure_credentials() may do a sync
        # HTTP token refresh, which would block the event loop.
        oauth = await anyio.to_thread.run_sync(ensure_credentials)
        model_name = _extract_model_name(str(request.url))

        body = json.loads(request.content) if request.content else {}
        cca_body = {
            "project": self._project_id,
            "model": model_name,
            "request": body,
            "userAgent": "rbtr",
        }

        cca_request = httpx.Request(
            "POST",
            f"{_CCA_GENERATE_URL}?alt=sse",
            headers={
                "Authorization": f"Bearer {oauth.access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                **_CCA_HEADERS,
            },
            content=json.dumps(cca_body).encode(),
        )

        response = await self._inner.handle_async_request(cca_request)

        if response.status_code >= 400:
            return response

        if not isinstance(response.stream, httpx.AsyncByteStream):
            raise TypeError("Expected async byte stream from async transport")
        stream: httpx.AsyncByteStream = response.stream
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_CCAStream(stream),
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


# ── Auth URL builder ─────────────────────────────────────────────────


def _build_auth_params(verifier: str) -> dict[str, str]:
    """Build query params for the Google authorize URL."""
    return {
        "client_id": _CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _REDIRECT_URI,
        "scope": _SCOPES,
        "code_challenge": make_challenge(verifier),
        "code_challenge_method": "S256",
        "state": verifier,
        "access_type": "offline",
        "prompt": "consent",
    }


# ── Token exchange and refresh ───────────────────────────────────────


def _make_oauth(
    data: dict[str, str | int],
    project_id: str,
    *,
    existing_refresh_token: str = "",
) -> OAuthCreds:
    """Build `OAuthCreds` from a token endpoint response.

    *existing_refresh_token* is preserved when the response doesn't
    include a new refresh token (Google typically omits it on refresh).
    """
    expires_in = data.get("expires_in")
    return OAuthCreds(
        access_token=str(data["access_token"]),
        refresh_token=str(data.get("refresh_token", "")) or existing_refresh_token,
        expires_at=time.time() + int(expires_in) if expires_in else None,
        project_id=project_id,
    )


def _exchange_code(code: str, verifier: str) -> OAuthCreds:
    """Exchange the authorization code for credentials."""
    data = token_request(
        _TOKEN_URL,
        {
            "client_id": _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": _REDIRECT_URI,
            "code_verifier": verifier,
        },
    )
    project_id = _discover_project(str(data["access_token"]))
    return _make_oauth(data, project_id)


def _refresh(oauth: OAuthCreds) -> OAuthCreds:
    """Use the refresh token to get a new access token."""
    data = token_request(
        _TOKEN_URL,
        {
            "client_id": _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
            "refresh_token": oauth.refresh_token,
            "grant_type": "refresh_token",
        },
    )
    return _make_oauth(data, oauth.project_id, existing_refresh_token=oauth.refresh_token)


# ── Automatic flow (localhost callback) ──────────────────────────────


def authenticate(cancel: threading.Event | None = None) -> OAuthCreds:
    """Run the full OAuth + PKCE flow with localhost callback."""
    verifier = make_verifier()
    params = _build_auth_params(verifier)
    code = run_oauth_flow(
        auth_url=_AUTH_URL,
        params=params,
        port=_REDIRECT_PORT,
        callback_path="/oauth2callback",
        expected_state=verifier,
        cancel=cancel,
    )
    return _exchange_code(code, verifier)


# ── Manual fallback (two-phase) ──────────────────────────────────────


def begin_login() -> tuple[str, PendingLogin]:
    """Phase 1: build authorize URL, open browser."""
    verifier = make_verifier()
    params = _build_auth_params(verifier)
    url = build_login_url(_AUTH_URL, params)
    return url, PendingLogin(code_verifier=verifier)


def complete_login(callback_url: str, pending: PendingLogin) -> OAuthCreds:
    """Phase 2: exchange the callback URL for credentials."""
    code, state = parse_callback_url(callback_url)
    if state and state != pending.code_verifier:
        raise RbtrError("OAuth state mismatch — possible CSRF attack.")
    return _exchange_code(code, pending.code_verifier)


# ── Credential persistence ──────────────────────────────────────────


def ensure_credentials() -> OAuthCreds:
    """Return valid credentials, refreshing if expired."""
    return _ensure_credentials("google", _refresh)


# ── Project discovery ────────────────────────────────────────────────


def _discover_project(access_token: str) -> str:
    """Discover or provision a Google Cloud project for Cloud Code Assist."""
    import os

    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
        "GOOGLE_CLOUD_PROJECT_ID"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    resp = httpx.post(
        f"{_CODE_ASSIST_URL}/v1internal:loadCodeAssist",
        json={
            "cloudaicompanionProject": env_project or "",
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        },
        headers=headers,
        timeout=30,
    )

    if resp.is_success:
        data = resp.json()
        if data.get("cloudaicompanionProject"):
            return data["cloudaicompanionProject"]
        if data.get("currentTier") and env_project:
            return env_project

    return _onboard_user(headers, env_project)


def _onboard_user(headers: dict[str, str], env_project: str | None) -> str:
    """Onboard the user to Cloud Code Assist (free tier)."""
    resp = httpx.post(
        f"{_CODE_ASSIST_URL}/v1internal:onboardUser",
        json={
            "tierId": "free-tier",
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        },
        headers=headers,
        timeout=60,
    )
    if not resp.is_success:
        raise RbtrError(
            f"Google Cloud onboarding failed ({resp.status_code}): {resp.text}. "
            "Try setting GOOGLE_CLOUD_PROJECT."
        )

    data = resp.json()
    if not data.get("done") and data.get("name"):
        project_id = _poll_for_project_id(data["name"], headers)
    else:
        response = data.get("response") or {}
        project = response.get("cloudaicompanionProject") or {}
        project_id = project.get("id") if isinstance(response, dict) else None

    if isinstance(project_id, str) and project_id:
        return project_id
    if env_project:
        return env_project

    raise RbtrError("Could not provision a Google Cloud project. Try setting GOOGLE_CLOUD_PROJECT.")


def _poll_for_project_id(name: str, headers: dict[str, str]) -> str | None:
    """Poll a long-running operation and extract the project ID."""
    for _ in range(30):
        time.sleep(5)
        resp = httpx.get(
            f"{_CODE_ASSIST_URL}/v1internal/{name}",
            headers=headers,
            timeout=30,
        )
        if not resp.is_success:
            raise RbtrError(f"Operation poll failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        if data.get("done"):
            response = data.get("response") or {}
            project = response.get("cloudaicompanionProject") or {}
            pid = project.get("id")
            return pid if isinstance(pid, str) else None
    raise RbtrError("Google Cloud provisioning timed out.")


# ── Provider ─────────────────────────────────────────────────────────


class GoogleProvider:
    """Google Gemini provider — OAuth or API key."""

    GENAI_ID = "google-gla"
    LABEL = "Google"

    def is_connected(self) -> bool:
        """Whether Gemini CLI OAuth credentials are available."""
        return oauth_is_set(creds.google)

    def list_models(self) -> list[str]:
        """Return known Gemini models from the `genai-prices` snapshot."""
        try:
            # Deferred: loads pricing snapshot from disk on first call.
            from genai_prices.data_snapshot import get_snapshot

            snapshot = get_snapshot()
            for p in snapshot.providers:
                if p.id == "google":
                    return sorted(
                        m.id
                        for m in p.models
                        if m.id.startswith("gemini-")
                        and "embedding" not in m.id
                        and "live" not in m.id
                    )
        except Exception:
            log.debug("Failed to load genai-prices snapshot for Google models", exc_info=True)
        return []

    def build_model(self, model_name: str) -> Model:
        """Build a Google model routed through Cloud Code Assist.

        Creates a google-genai Client with `vertexai=True` so the SDK
        formats requests in the standard Gemini API structure.  The
        actual HTTP is intercepted by `_CCATransport` which wraps
        the body and redirects to `cloudcode-pa.googleapis.com`.
        """
        # Deferred: google SDK is heavy; only load when this provider is used.
        from google.genai.client import Client
        from google.genai.types import HttpOptions
        from google.oauth2.credentials import Credentials
        from pydantic_ai.models.google import GoogleModel as _GoogleModel
        from pydantic_ai.providers.google import GoogleProvider as _GoogleProvider

        oauth = ensure_credentials()
        transport = _CCATransport(oauth.project_id)
        http_client = httpx.AsyncClient(transport=transport)

        # Credentials with no expiry → SDK never attempts its own refresh.
        # The transport calls `ensure_credentials()` per-request instead.
        dummy_creds = Credentials(  # type: ignore[no-untyped-call]  # google-auth stubs incomplete
            token="cca-transport-handles-auth",  # noqa: S106  # not a real secret
        )
        client = Client(
            vertexai=True,
            credentials=dummy_creds,
            project=oauth.project_id,
            http_options=HttpOptions(httpx_async_client=http_client),
        )
        prov = _GoogleProvider(client=client)
        return _GoogleModel(model_name, provider=prov)

    def model_settings(
        self, model_id: str, model: Model, effort: ThinkingEffort
    ) -> ModelSettings | None:
        """Google models don't support reasoning effort parameters."""
        return None

    def system_instructions(self, model_id: str) -> str | None:
        return None

    def context_window(self, model_id: str) -> int | None:
        """Look up context window from `genai-prices`."""
        from rbtr.providers.shared import genai_prices_context_window

        return genai_prices_context_window(self.GENAI_ID, model_id)


provider = GoogleProvider()
