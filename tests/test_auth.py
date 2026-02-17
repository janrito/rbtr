"""Tests for rbtr.github.auth."""

import threading
from pathlib import Path

import pytest

from rbtr import RbtrError
from rbtr.creds import creds
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


def _mock_httpx_response(json_data: dict[str, str]) -> object:
    class _Response:
        def json(self) -> dict[str, str]:
            return json_data

        def raise_for_status(self) -> None:
            pass

    return _Response()


def _patch_httpx_client(mocker, responses):
    mock_cls = mocker.patch("rbtr.github.auth.httpx.Client")
    mock_cls.return_value.__enter__ = lambda s: s
    mock_cls.return_value.__exit__ = lambda s, *a: None
    if isinstance(responses, list):
        mock_cls.return_value.post.side_effect = responses
    else:
        mock_cls.return_value.post.return_value = responses
    return mock_cls


def test_poll_returns_token_on_success(mocker) -> None:
    _patch_httpx_client(mocker, _mock_httpx_response({"access_token": "ghp_ok"}))
    assert auth.poll_for_token("device123", 0) == "ghp_ok"


def test_poll_retries_on_authorization_pending(mocker) -> None:
    _patch_httpx_client(
        mocker,
        [
            _mock_httpx_response({"error": "authorization_pending"}),
            _mock_httpx_response({"access_token": "ghp_after_wait"}),
        ],
    )
    assert auth.poll_for_token("device123", 0) == "ghp_after_wait"


def test_poll_raises_on_expired_token(mocker) -> None:
    _patch_httpx_client(mocker, _mock_httpx_response({"error": "expired_token"}))
    with pytest.raises(RbtrError, match="expired"):
        auth.poll_for_token("device123", 0)


def test_poll_raises_on_access_denied(mocker) -> None:
    _patch_httpx_client(mocker, _mock_httpx_response({"error": "access_denied"}))
    with pytest.raises(RbtrError, match="cancelled"):
        auth.poll_for_token("device123", 0)


def test_poll_raises_on_cancel(mocker) -> None:
    cancel = threading.Event()
    cancel.set()
    _patch_httpx_client(
        mocker,
        _mock_httpx_response({"error": "authorization_pending"}),
    )
    with pytest.raises(RbtrError, match="Cancelled"):
        auth.poll_for_token("device123", 5, cancel=cancel)
