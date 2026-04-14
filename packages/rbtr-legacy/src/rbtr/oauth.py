"""Shared OAuth / PKCE utilities for provider modules.

Centralises the trust-critical paths — credential lifecycle, token
endpoint communication, callback server, and PKCE — so they can be
audited in one place.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from rbtr.config import config
from rbtr.exceptions import PortBusyError, RbtrError

if TYPE_CHECKING:
    from rbtr.creds import OAuthCreds

# Type alias for parsed token-endpoint JSON responses.
# Standard OAuth fields are strings or ints (expires_in).
type TokenData = dict[str, str | int]


# ── Shared data ──────────────────────────────────────────────────────


@dataclass
class PendingLogin:
    """State kept between the two phases of an OAuth login flow.

    All OAuth providers store this between `begin_login` (phase 1)
    and `complete_login` (phase 2).  `state` is used by providers
    that generate a separate random state parameter (e.g. ChatGPT);
    providers that use the verifier as state leave it empty.
    """

    code_verifier: str
    state: str = ""


# ── Obfuscation ──────────────────────────────────────────────────────


def deobfuscate(encoded: str) -> str:
    """Decode a base64-encoded constant (client IDs, secrets).

    All provider modules store OAuth client identifiers as base64 to
    keep them out of casual `grep` results.  This function documents
    that convention and provides a single decode path.
    """
    return base64.b64decode(encoded).decode()


# ── PKCE ─────────────────────────────────────────────────────────────


def make_verifier() -> str:
    """Generate a cryptographically random PKCE code verifier (43-128 chars)."""
    return secrets.token_urlsafe(64)[:128]


def make_challenge(verifier: str) -> str:
    """Derive a S256 code challenge from the verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


# ── Credential checks ───────────────────────────────────────────────


def oauth_is_set(oauth: OAuthCreds) -> bool:
    """Return whether the token set contains usable credentials.

    An access token exists AND is either unexpired or refreshable.
    """
    if not oauth.access_token:
        return False
    if oauth.refresh_token:
        return True
    return not oauth_expired(oauth)


def oauth_expired(oauth: OAuthCreds) -> bool:
    """Return whether the access token has expired (or is about to)."""
    if oauth.expires_at is None:
        return False
    return time.time() >= oauth.expires_at - config.oauth.refresh_buffer_seconds


# ── Localhost callback server ────────────────────────────────────────


class OAuthCallbackServer(http.server.HTTPServer):
    """Receives an OAuth redirect on `localhost:<port><path>`.

    After `handle_request()` completes, inspect `code`, `state`,
    and `error` to determine what happened.
    """

    code: str | None
    state: str | None
    error: str | None

    def __init__(self, port: int, callback_path: str) -> None:
        self._callback_path = callback_path
        super().__init__(("127.0.0.1", port), _OAuthCallbackHandler)
        self.code = None
        self.state = None
        self.error = None


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    server: OAuthCallbackServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != self.server._callback_path:
            self._respond(404, "Not found")
            return

        qs = parse_qs(parsed.query)

        errors = qs.get("error", [])
        if errors:
            self.server.error = errors[0]
            self._respond(400, f"Authentication failed: {errors[0]}")
            return

        codes = qs.get("code", [])
        if not codes:
            self.server.error = "Missing authorization code"
            self._respond(400, "Missing authorization code")
            return

        self.server.code = codes[0]
        states = qs.get("state", [])
        self.server.state = states[0] if states else None
        self._respond(200, "Authentication successful. Return to your terminal to continue.")

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


def start_callback_server(port: int, callback_path: str) -> OAuthCallbackServer:
    """Create and return a callback server, raising `PortBusyError` on failure."""
    try:
        return OAuthCallbackServer(port, callback_path)
    except OSError as e:
        raise PortBusyError(f"Port {port} is busy: {e}") from e


# ── Automatic OAuth flow ────────────────────────────────────────────


def run_oauth_flow(
    *,
    auth_url: str,
    params: dict[str, str],
    port: int,
    callback_path: str,
    expected_state: str,
    cancel: threading.Event | None = None,
    timeout: float | None = None,
) -> str:
    """Run the OAuth PKCE flow with a localhost callback, returning the code.

    Opens the browser, starts a callback server, waits for the
    redirect, validates state, and returns the authorization code.
    Raises `PortBusyError` if the port is taken.
    """
    url = f"{auth_url}?{urlencode(params)}"
    server = start_callback_server(port, callback_path)
    try:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

        server.timeout = 0.5
        deadline = time.time() + timeout if timeout else None
        while server.code is None and server.error is None:
            if cancel is not None and cancel.is_set():
                raise RbtrError("Cancelled.")
            if deadline and time.time() > deadline:
                raise RbtrError("Authorization timed out.")
            server.handle_request()
    finally:
        server.server_close()

    if server.error or server.code is None:
        raise RbtrError(f"Authentication failed: {server.error or 'no code received'}")
    if server.state != expected_state:
        raise RbtrError("OAuth state mismatch — possible CSRF attack.")
    return server.code


def build_login_url(auth_url: str, params: dict[str, str]) -> str:
    """Build the authorize URL and open it in the browser.

    Used by the manual fallback flow (`begin_login`).
    """
    url = f"{auth_url}?{urlencode(params)}"
    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    return url


def parse_callback_url(raw: str) -> tuple[str, str]:
    """Extract `(code, state)` from a callback URL or pasted string.

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
            state = (qs.get("state", [""]) or [""])[0]
            return code, state
    except ValueError:
        pass

    # Bare code (no URL structure)
    if "=" not in raw and "&" not in raw:
        return raw, ""

    # Query-string format: code=...&state=...
    qs = parse_qs(raw)
    code = qs.get("code", [None])[0]
    if code:
        state = (qs.get("state", [""]) or [""])[0]
        return code, state

    raise RbtrError(
        "Could not extract authorization code. Paste the full redirect URL from your browser."
    )


# ── Token endpoint ───────────────────────────────────────────────────


def token_request(
    url: str,
    body: dict[str, str],
    *,
    as_json: bool = False,
) -> TokenData:
    """POST to an OAuth token endpoint, returning parsed JSON.

    Centralises timeout, error handling, and status checking for all
    token exchange and refresh calls.
    """
    if as_json:
        resp = httpx.post(url, json=body, timeout=30)
    else:
        resp = httpx.post(url, data=body, timeout=30)
    if not resp.is_success:
        raise RbtrError(f"Token request to {url} failed ({resp.status_code}): {resp.text}")
    result: TokenData = resp.json()
    return result


# ── Credential lifecycle ─────────────────────────────────────────────


def ensure_credentials(
    field: str,
    refresh: Callable[[OAuthCreds], OAuthCreds],
) -> OAuthCreds:
    """Return valid OAuth credentials, refreshing if expired.

    *field* is the attribute name on `creds` (e.g. `"claude"`),
    used both for reading/writing credentials and in error messages.

    When refresh fails (e.g. the refresh token expired or was
    revoked), the stale credentials are cleared so
    `is_connected()` returns `False` and the user sees the
    disconnected state immediately.

    Raises `RbtrError` if not connected or refresh fails.
    """
    from rbtr.creds import creds  # deferred: avoid circular import at module level

    oauth: OAuthCreds = getattr(creds, field)
    if not oauth_is_set(oauth):
        raise RbtrError(f"Not connected. Use /connect {field}.")

    if not oauth_expired(oauth):
        return oauth

    if not oauth.refresh_token:
        raise RbtrError(f"Session expired. Use /connect {field} to re-authenticate.")

    try:
        refreshed = refresh(oauth)
    except RbtrError:
        # Refresh token is invalid or revoked — clear stale
        # credentials so `is_connected()` reflects reality.
        from rbtr.creds import OAuthCreds as OAuthCredsModel

        creds.update(**{field: OAuthCredsModel()})
        raise RbtrError(
            f"Session expired (refresh token rejected). Use /connect {field} to re-authenticate."
        ) from None

    creds.update(**{field: refreshed})
    return refreshed
