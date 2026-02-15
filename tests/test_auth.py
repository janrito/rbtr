"""Tests for rbtr.github.auth."""

import stat
from pathlib import Path

import pytest

from rbtr import RbtrError
from rbtr.github import auth

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def token_path(tmp_path: Path, mocker):
    """Patch TOKEN_PATH to a temporary file and return its Path."""
    p = tmp_path / "token"
    mocker.patch.object(auth, "TOKEN_PATH", p)
    return p


# ── load_token ───────────────────────────────────────────────────────


def test_load_token_returns_stored(token_path: Path) -> None:
    token_path.write_text("ghp_stored123")
    assert auth.load_token() == "ghp_stored123"


def test_load_token_returns_none_when_no_file(token_path: Path) -> None:
    assert auth.load_token() is None


def test_load_token_returns_none_when_empty(token_path: Path) -> None:
    token_path.write_text("  \n")
    assert auth.load_token() is None


# ── clear_token ──────────────────────────────────────────────────────


def test_clear_token_removes_file(token_path: Path) -> None:
    token_path.write_text("ghp_old")
    auth.clear_token()
    assert not token_path.exists()


def test_clear_token_no_error_when_missing(token_path: Path) -> None:
    auth.clear_token()  # should not raise


# ── save_token ───────────────────────────────────────────────────────


def test_save_token_creates_directory_and_file(tmp_path: Path, mocker) -> None:
    token_path = tmp_path / "nested" / "dir" / "token"
    mocker.patch.object(auth, "TOKEN_PATH", token_path)
    auth.save_token("ghp_test")
    assert token_path.read_text() == "ghp_test"
    assert token_path.stat().st_mode & 0o777 == stat.S_IRUSR | stat.S_IWUSR


# ── poll_for_token ───────────────────────────────────────────────────


def _mock_httpx_response(json_data: dict[str, str]) -> object:
    """Create a mock httpx response with a .json() method and .raise_for_status()."""

    class _Response:
        def json(self) -> dict[str, str]:
            return json_data

        def raise_for_status(self) -> None:
            pass

    return _Response()


def _patch_httpx_client(mocker, responses):
    """Patch httpx.Client to return canned responses from .post()."""
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
