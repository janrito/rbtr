"""Tests for rbtr.providers.claude — PKCE, credential persistence, token refresh."""

import base64
import hashlib
import time
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.creds import OAuthCreds, creds
from rbtr.exceptions import RbtrError
from rbtr.oauth import make_challenge, make_verifier, oauth_is_set
from rbtr.providers.claude import (
    _AUTHORIZE_URL,
    _CLIENT_ID,
    begin_login,
    ensure_credentials,
)

# ── PKCE ─────────────────────────────────────────────────────────────


def test_verifier_length() -> None:
    v = make_verifier()
    assert 43 <= len(v) <= 128


def test_verifier_is_url_safe() -> None:
    v = make_verifier()
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    assert set(v).issubset(allowed)


def test_challenge_is_s256() -> None:
    verifier = "test-verifier-value"
    challenge = make_challenge(verifier)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert challenge == expected


# ── begin_login ──────────────────────────────────────────────────────


def test_begin_login_returns_url_and_pending(mocker: MockerFixture) -> None:
    mocker.patch("rbtr.oauth.webbrowser.open")
    url, pending = begin_login()
    assert url.startswith(_AUTHORIZE_URL)
    assert _CLIENT_ID in url
    assert "code_challenge_method=S256" in url
    assert "response_type=code" in url
    assert pending.code_verifier


def test_begin_login_opens_browser(mocker: MockerFixture) -> None:
    mock_open = mocker.patch("rbtr.oauth.webbrowser.open")
    url, _ = begin_login()

    time.sleep(0.1)
    mock_open.assert_called_once_with(url)


# ── Credential persistence ───────────────────────────────────────────


def _make_oauth(**overrides: object) -> OAuthCreds:
    defaults: dict = {
        "access_token": "test-access-token",
        "refresh_token": "test-refresh-token",
        "expires_at": time.time() + 3600,
    }
    return OAuthCreds(**{**defaults, **overrides})


def test_save_load_clear_credentials(creds_path: Path) -> None:
    assert not oauth_is_set(creds.claude)

    oauth = _make_oauth()
    creds.update(claude=oauth)

    assert oauth_is_set(creds.claude)
    assert creds.claude.access_token == "test-access-token"
    assert creds.claude.refresh_token == "test-refresh-token"
    assert creds.claude.expires_at is not None
    assert creds.claude.expires_at > time.time()

    creds.update(claude=OAuthCreds())
    assert not oauth_is_set(creds.claude)


def test_load_credentials_handles_empty_token(creds_path: Path) -> None:
    creds.update(claude=OAuthCreds(access_token="", refresh_token="", expires_at=None))
    assert not oauth_is_set(creds.claude)


# ── ensure_credentials ───────────────────────────────────────────────


def _store_oauth(
    creds_path: Path,
    *,
    access_token: str = "tok",
    refresh_token: str = "ref",
    expires_at: float | None = None,
) -> None:
    creds.update(
        claude=OAuthCreds(
            access_token=access_token, refresh_token=refresh_token, expires_at=expires_at
        )
    )


def test_ensure_returns_creds_when_not_expired(creds_path: Path) -> None:
    _store_oauth(creds_path, expires_at=time.time() + 3600)
    oauth = ensure_credentials()
    assert oauth.access_token == "tok"


def test_ensure_returns_creds_when_no_expiry(creds_path: Path) -> None:
    _store_oauth(creds_path, expires_at=None)
    oauth = ensure_credentials()
    assert oauth.access_token == "tok"


def test_ensure_refreshes_expired_token(creds_path: Path, mocker: MockerFixture) -> None:
    _store_oauth(
        creds_path, access_token="old-tok", refresh_token="ref-tok", expires_at=time.time() - 100
    )
    refreshed = _make_oauth(access_token="new-tok", refresh_token="new-ref")
    mocker.patch("rbtr.providers.claude._refresh", return_value=refreshed)

    oauth = ensure_credentials()
    assert oauth.access_token == "new-tok"
    assert creds.claude.access_token == "new-tok"


def test_ensure_refreshes_within_buffer(creds_path: Path, mocker: MockerFixture) -> None:
    _store_oauth(creds_path, expires_at=time.time() + config.oauth.refresh_buffer_seconds - 10)
    refreshed = _make_oauth(access_token="refreshed")
    mocker.patch("rbtr.providers.claude._refresh", return_value=refreshed)

    oauth = ensure_credentials()
    assert oauth.access_token == "refreshed"


def test_ensure_raises_when_no_credentials(creds_path: Path) -> None:
    with pytest.raises(RbtrError, match="Not connected"):
        ensure_credentials()


def test_ensure_raises_when_expired_no_refresh_token(creds_path: Path) -> None:
    _store_oauth(creds_path, refresh_token="", expires_at=time.time() - 100)
    with pytest.raises(RbtrError, match="Not connected"):
        ensure_credentials()


def test_ensure_clears_creds_on_refresh_failure(creds_path: Path, mocker: MockerFixture) -> None:
    """When the refresh token is rejected (e.g. expired or revoked),
    stale credentials are cleared so `is_connected()` reflects reality."""
    _store_oauth(creds_path, expires_at=time.time() - 100)
    mocker.patch(
        "rbtr.providers.claude._refresh",
        side_effect=RbtrError("Token request failed (400): invalid_grant"),
    )

    with pytest.raises(RbtrError, match="Session expired"):
        ensure_credentials()

    # Credentials must be wiped — is_connected should return False.
    assert not oauth_is_set(creds.claude)
    assert creds.claude.access_token == ""
    assert creds.claude.refresh_token == ""
