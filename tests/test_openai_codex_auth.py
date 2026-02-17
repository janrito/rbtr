"""Tests for rbtr.providers.openai_codex — PKCE, JWT, code parsing, credentials."""

import base64
import hashlib
import json
import time
from pathlib import Path

import pytest

from rbtr import RbtrError
from rbtr.config import config
from rbtr.creds import OAuthCreds, creds
from rbtr.oauth import make_challenge, make_verifier, oauth_is_set
from rbtr.providers.openai_codex import (
    PendingLogin,
    _read_account_id,
    begin_login,
    complete_login,
    ensure_credentials,
    parse_callback_url,
)

# ── PKCE ─────────────────────────────────────────────────────────────


def test_verifier_is_url_safe() -> None:
    v = make_verifier()
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    assert set(v).issubset(allowed)
    assert len(v) >= 32


def test_challenge_is_s256() -> None:
    verifier = "test-verifier-value"
    challenge = make_challenge(verifier)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert challenge == expected


# ── JWT extraction ───────────────────────────────────────────────────


def _make_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.sig"


def test_read_account_id_valid() -> None:
    jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_abc123"}})
    assert _read_account_id(jwt) == "acct_abc123"


def test_read_account_id_missing_claim() -> None:
    jwt = _make_jwt({"sub": "user123"})
    with pytest.raises(RbtrError, match="account ID"):
        _read_account_id(jwt)


def test_read_account_id_not_a_jwt() -> None:
    with pytest.raises(RbtrError, match="account ID"):
        _read_account_id("not-a-jwt")


# ── parse_callback_url ───────────────────────────────────────────────


def test_parse_callback_full_url() -> None:
    url = "http://localhost:1455/auth/callback?code=abc123&state=xyz"
    assert parse_callback_url(url) == "abc123"


def test_parse_callback_bare_code() -> None:
    assert parse_callback_url("abc123") == "abc123"


def test_parse_callback_strips_whitespace() -> None:
    assert parse_callback_url("  abc123  \n") == "abc123"


def test_parse_callback_empty_raises() -> None:
    with pytest.raises(RbtrError, match="Empty input"):
        parse_callback_url("")


def test_parse_callback_query_string() -> None:
    assert parse_callback_url("code=abc&state=xyz") == "abc"


# ── begin_login ──────────────────────────────────────────────────────


def test_begin_login_returns_url_and_pending(mocker) -> None:
    mocker.patch("rbtr.providers.openai_codex.webbrowser.open")
    url, pending = begin_login()
    assert url.startswith(config.providers.chatgpt.authorize_url)
    assert config.providers.chatgpt.client_id in url
    assert "code_challenge_method=S256" in url
    assert pending.code_verifier
    assert pending.state


# ── Credential persistence ───────────────────────────────────────────


def _make_oauth(**overrides: object) -> OAuthCreds:
    defaults: dict = {
        "access_token": "tok",
        "refresh_token": "ref",
        "expires_at": time.time() + 3600,
        "account_id": "acct_123",
    }
    return OAuthCreds(**{**defaults, **overrides})


def test_save_load_clear_credentials(creds_path: Path) -> None:
    assert not oauth_is_set(creds.chatgpt)

    oauth = _make_oauth()
    creds.update(chatgpt=oauth)

    assert oauth_is_set(creds.chatgpt)
    assert creds.chatgpt.access_token == "tok"
    assert creds.chatgpt.refresh_token == "ref"
    assert creds.chatgpt.account_id == "acct_123"
    assert creds.chatgpt.expires_at is not None
    assert creds.chatgpt.expires_at > time.time()

    creds.update(chatgpt=OAuthCreds())
    assert not oauth_is_set(creds.chatgpt)


def test_load_credentials_handles_empty_token(creds_path: Path) -> None:
    creds.update(chatgpt=OAuthCreds(access_token="", refresh_token="", expires_at=0, account_id=""))
    assert not oauth_is_set(creds.chatgpt)


# ── ensure_credentials ───────────────────────────────────────────────


def _store_oauth(
    creds_path: Path,
    *,
    access_token: str = "tok",
    refresh_token: str = "ref",
    expires_at: float | None = None,
    account_id: str = "acct_1",
) -> None:
    creds.update(
        chatgpt=OAuthCreds(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at if expires_at is not None else time.time() + 3600,
            account_id=account_id,
        )
    )


def test_ensure_returns_creds_when_not_expired(creds_path: Path) -> None:
    _store_oauth(creds_path)
    oauth = ensure_credentials()
    assert oauth.access_token == "tok"
    assert oauth.account_id == "acct_1"


def test_ensure_refreshes_expired_token(creds_path: Path, mocker) -> None:
    _store_oauth(creds_path, access_token="old-tok", expires_at=time.time() - 100)
    new_jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_new"}})
    refreshed = OAuthCreds(
        access_token=new_jwt,
        refresh_token="new-ref",
        expires_at=time.time() + 3600,
        account_id="acct_new",
    )
    mocker.patch("rbtr.providers.openai_codex._refresh", return_value=refreshed)

    oauth = ensure_credentials()
    assert oauth.access_token == new_jwt
    assert oauth.account_id == "acct_new"


def test_ensure_raises_when_no_credentials(creds_path: Path) -> None:
    with pytest.raises(RbtrError, match="Not connected"):
        ensure_credentials()


def test_ensure_raises_when_expired_no_refresh(creds_path: Path) -> None:
    _store_oauth(creds_path, refresh_token="", expires_at=time.time() - 100)
    with pytest.raises(RbtrError, match="expired"):
        ensure_credentials()


# ── complete_login ───────────────────────────────────────────────────


def test_complete_login_exchanges_code(mocker) -> None:
    oauth = _make_oauth()
    mocker.patch("rbtr.providers.openai_codex._exchange_code", return_value=oauth)

    pending = PendingLogin(code_verifier="verifier", state="st")
    result = complete_login("mycode", pending)
    assert result.access_token == "tok"
    assert result.account_id == "acct_123"
