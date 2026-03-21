"""Tests for rbtr.github.auth."""

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from rbtr.creds import creds
from rbtr.exceptions import RbtrError
from rbtr.github import auth

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


@pytest.fixture
def httpx_client(mocker: MockerFixture) -> MagicMock:
    """Mock httpx.Client used by poll_for_token.

    Tests configure `httpx_client.post.return_value` or
    `.post.side_effect` with MagicMock responses.
    """
    mock_cls = mocker.patch("rbtr.github.auth.httpx.Client")
    client = MagicMock()
    mock_cls.return_value.__enter__ = lambda s: client
    mock_cls.return_value.__exit__ = lambda s, *a: None
    return client


def _response(data: dict[str, str]) -> MagicMock:
    """Build a mock httpx response with JSON data."""
    r = MagicMock()
    r.json.return_value = data
    return r


def test_poll_returns_token_on_success(httpx_client: MagicMock) -> None:
    httpx_client.post.return_value = _response({"access_token": "ghp_ok"})
    assert auth.poll_for_token("device123", 0) == "ghp_ok"


def test_poll_retries_on_authorization_pending(httpx_client: MagicMock) -> None:
    httpx_client.post.side_effect = [
        _response({"error": "authorization_pending"}),
        _response({"access_token": "ghp_after_wait"}),
    ]
    assert auth.poll_for_token("device123", 0) == "ghp_after_wait"


def test_poll_raises_on_expired_token(httpx_client: MagicMock) -> None:
    httpx_client.post.return_value = _response({"error": "expired_token"})
    with pytest.raises(RbtrError, match="expired"):
        auth.poll_for_token("device123", 0)


def test_poll_raises_on_access_denied(httpx_client: MagicMock) -> None:
    httpx_client.post.return_value = _response({"error": "access_denied"})
    with pytest.raises(RbtrError, match="cancelled"):
        auth.poll_for_token("device123", 0)


def test_poll_raises_on_cancel(httpx_client: MagicMock) -> None:
    cancel = threading.Event()
    cancel.set()
    httpx_client.post.return_value = _response({"error": "authorization_pending"})
    with pytest.raises(RbtrError, match="Cancelled"):
        auth.poll_for_token("device123", 5, cancel=cancel)
