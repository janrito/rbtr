"""Tests for rbtr.github.auth."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from rbtr.creds import creds
from rbtr.exceptions import RbtrError
from rbtr.github import auth

_OAUTH_URL = "https://github.com/login/oauth/access_token"

# ── GitHub token via creds ───────────────────────────────────────────


def test_github_token_roundtrip(creds_path: Path) -> None:
    creds.update(github_token="ghp_stored123")
    assert creds.github_token == "ghp_stored123"


def test_github_token_empty_by_default(creds_path: Path) -> None:
    assert creds.github_token == ""


def test_github_token_clear(creds_path: Path) -> None:
    creds.update(github_token="ghp_old")
    creds.update(github_token="")
    assert creds.github_token == ""


# ── poll_for_token ───────────────────────────────────────────────────


def test_poll_returns_token_on_success(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=_OAUTH_URL, json={"access_token": "ghp_ok"})
    assert auth.poll_for_token("device123", 0) == "ghp_ok"


def test_poll_retries_on_authorization_pending(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=_OAUTH_URL, json={"error": "authorization_pending"})
    httpx_mock.add_response(url=_OAUTH_URL, json={"access_token": "ghp_after_wait"})
    assert auth.poll_for_token("device123", 0) == "ghp_after_wait"


def test_poll_raises_on_expired_token(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=_OAUTH_URL, json={"error": "expired_token"})
    with pytest.raises(RbtrError, match="expired"):
        auth.poll_for_token("device123", 0)


def test_poll_raises_on_access_denied(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=_OAUTH_URL, json={"error": "access_denied"})
    with pytest.raises(RbtrError, match="cancelled"):
        auth.poll_for_token("device123", 0)


def test_poll_raises_on_cancel() -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RbtrError, match="Cancelled"):
        auth.poll_for_token("device123", 5, cancel=cancel)
