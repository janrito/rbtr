"""Tests for rbtr.creds — singleton credential storage.

Uses a data-first approach: shared credential data constants
represent realistic, semantically distinct credential sets so
tests verify roundtrip, isolation, and persistence against
inspectable named data — not anonymous inline strings.
"""

import stat
from pathlib import Path

import pytest

from rbtr.config import config
from rbtr.creds import OAuthCreds, creds

# ── Shared test data ─────────────────────────────────────────────────
#
# Three distinct credential sets covering every slot type:
# OAuth (Claude, ChatGPT), API key (OpenAI), token (GitHub),
# and endpoint keys.

_CLAUDE_OAUTH = OAuthCreds(
    access_token="claude-bearer-tok",
    refresh_token="claude-refresh",
    expires_at=1000.0,
)

_CHATGPT_OAUTH = OAuthCreds(
    access_token="chatgpt-bearer-tok",
    refresh_token="chatgpt-refresh",
    expires_at=2000.0,
    account_id="acct_42",
)

_GITHUB_TOKEN = "ghp_abc123"
_OPENAI_KEY = "sk-test-key-789"
_ENDPOINT_KEYS = {"alpha": "key-alpha", "bravo": "key-bravo"}

# Full credential set — every slot populated.
_ALL_CREDS: dict = {
    "github_token": _GITHUB_TOKEN,
    "openai_api_key": _OPENAI_KEY,
    "claude": _CLAUDE_OAUTH,
    "chatgpt": _CHATGPT_OAUTH,
    "endpoint_keys": _ENDPOINT_KEYS,
}


def _seed_all(creds_path: Path) -> None:
    """Populate every credential slot with known test data."""
    creds.update(**_ALL_CREDS)


# ── Basics ───────────────────────────────────────────────────────────


def test_defaults_when_no_file(creds_path: Path) -> None:
    assert creds.github_token == ""
    assert creds.claude.access_token == ""
    assert creds.openai_api_key == ""
    assert creds.endpoint_keys == {}


def test_update_persists_and_reloads(creds_path: Path) -> None:
    creds.update(github_token=_GITHUB_TOKEN)
    assert creds.github_token == _GITHUB_TOKEN
    assert creds_path.exists()


def test_update_sets_0600_permissions(creds_path: Path) -> None:
    creds.update(github_token="x")
    assert stat.S_IMODE(creds_path.stat().st_mode) == 0o600


def test_update_creates_parent_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nested = tmp_path / "nested" / "dir"
    monkeypatch.setenv("RBTR_USER_DIR", str(nested))
    config.reload()
    creds.reload()
    creds.update(github_token="x")
    assert (nested / "creds.toml").exists()


def test_identity_preserved_after_update(creds_path: Path) -> None:
    original = creds
    creds.update(github_token="x")
    assert creds is original


# ── OAuth slots ──────────────────────────────────────────────────────


def test_oauth_roundtrip(creds_path: Path) -> None:
    assert not creds.claude.access_token

    creds.update(claude=_CLAUDE_OAUTH)

    assert creds.claude.access_token == _CLAUDE_OAUTH.access_token
    assert creds.claude.refresh_token == _CLAUDE_OAUTH.refresh_token
    assert creds.claude.expires_at == _CLAUDE_OAUTH.expires_at


def test_oauth_slots_are_independent(creds_path: Path) -> None:
    creds.update(claude=_CLAUDE_OAUTH, chatgpt=_CHATGPT_OAUTH)

    assert creds.claude.access_token == _CLAUDE_OAUTH.access_token
    assert creds.chatgpt.access_token == _CHATGPT_OAUTH.access_token
    assert creds.chatgpt.account_id == _CHATGPT_OAUTH.account_id


def test_clear_oauth_slot(creds_path: Path) -> None:
    creds.update(claude=_CLAUDE_OAUTH, chatgpt=_CHATGPT_OAUTH)

    creds.update(claude=OAuthCreds())

    assert not creds.claude.access_token
    assert creds.chatgpt.access_token == _CHATGPT_OAUTH.access_token


# ── Simple string fields ────────────────────────────────────────────


def test_openai_api_key_roundtrip(creds_path: Path) -> None:
    assert not creds.openai_api_key
    creds.update(openai_api_key=_OPENAI_KEY)
    assert creds.openai_api_key == _OPENAI_KEY


def test_github_token_roundtrip(creds_path: Path) -> None:
    creds.update(github_token=_GITHUB_TOKEN)
    assert creds.github_token == _GITHUB_TOKEN


# ── Endpoint keys ────────────────────────────────────────────────────


def test_endpoint_key_roundtrip(creds_path: Path) -> None:
    assert creds.endpoint_keys.get("alpha", "") == ""
    creds.update(endpoint_keys=_ENDPOINT_KEYS)
    assert creds.endpoint_keys["alpha"] == _ENDPOINT_KEYS["alpha"]


def test_endpoint_keys_are_independent(creds_path: Path) -> None:
    creds.update(endpoint_keys=_ENDPOINT_KEYS)
    assert creds.endpoint_keys["alpha"] == "key-alpha"
    assert creds.endpoint_keys["bravo"] == "key-bravo"


# ── Cross-slot isolation ────────────────────────────────────────────


def test_update_preserves_other_slots(creds_path: Path) -> None:
    _seed_all(creds_path)

    # Overwrite one field.
    creds.update(openai_api_key="sk-new")

    assert creds.github_token == _GITHUB_TOKEN
    assert creds.openai_api_key == "sk-new"
    assert creds.claude.access_token == _CLAUDE_OAUTH.access_token
    assert creds.endpoint_keys["alpha"] == _ENDPOINT_KEYS["alpha"]
