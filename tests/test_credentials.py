"""Tests for rbtr.creds — singleton credential storage."""

import stat
from pathlib import Path

from rbtr.creds import OAuthCreds, creds

# ── Basics ───────────────────────────────────────────────────────────


def test_defaults_when_no_file(creds_path: Path) -> None:
    assert creds.github_token == ""
    assert creds.claude.access_token == ""
    assert creds.openai_api_key == ""
    assert creds.endpoint_keys == {}


def test_update_persists_and_reloads(creds_path: Path) -> None:
    creds.update(github_token="ghp_abc")
    assert creds.github_token == "ghp_abc"
    assert creds_path.exists()


def test_update_sets_0600_permissions(creds_path: Path) -> None:
    creds.update(github_token="x")
    assert stat.S_IMODE(creds_path.stat().st_mode) == 0o600


def test_update_creates_parent_dirs(tmp_path: Path, monkeypatch) -> None:
    from rbtr.creds import Creds

    path = tmp_path / "nested" / "dir" / "creds.toml"
    monkeypatch.setattr("rbtr.creds.CREDS_PATH", path)
    monkeypatch.setitem(Creds.model_config, "toml_file", str(path))
    creds.__init__()  # type: ignore[misc]
    creds.update(github_token="x")
    assert path.exists()


def test_identity_preserved_after_update(creds_path: Path) -> None:
    original = creds
    creds.update(github_token="x")
    assert creds is original


# ── OAuth slots ──────────────────────────────────────────────────────


def test_oauth_roundtrip(creds_path: Path) -> None:
    assert not creds.claude.access_token

    creds.update(claude=OAuthCreds(access_token="tok", refresh_token="ref", expires_at=1000.0))

    assert creds.claude.access_token == "tok"
    assert creds.claude.refresh_token == "ref"
    assert creds.claude.expires_at == 1000.0


def test_oauth_slots_are_independent(creds_path: Path) -> None:
    creds.update(
        claude=OAuthCreds(access_token="claude-tok"),
        chatgpt=OAuthCreds(access_token="chatgpt-tok", account_id="acct"),
    )

    assert creds.claude.access_token == "claude-tok"
    assert creds.chatgpt.access_token == "chatgpt-tok"
    assert creds.chatgpt.account_id == "acct"


def test_clear_oauth_slot(creds_path: Path) -> None:
    creds.update(
        claude=OAuthCreds(access_token="tok"),
        chatgpt=OAuthCreds(access_token="other"),
    )

    creds.update(claude=OAuthCreds())

    assert not creds.claude.access_token
    assert creds.chatgpt.access_token == "other"


# ── Simple string fields ────────────────────────────────────────────


def test_openai_api_key_roundtrip(creds_path: Path) -> None:
    assert not creds.openai_api_key
    creds.update(openai_api_key="sk-123")
    assert creds.openai_api_key == "sk-123"


def test_github_token_roundtrip(creds_path: Path) -> None:
    creds.update(github_token="ghp_abc")
    assert creds.github_token == "ghp_abc"


# ── Endpoint keys ────────────────────────────────────────────────────


def test_endpoint_key_roundtrip(creds_path: Path) -> None:
    assert creds.endpoint_keys.get("test", "") == ""
    creds.update(endpoint_keys={"test": "sk-ep"})
    assert creds.endpoint_keys["test"] == "sk-ep"


def test_endpoint_keys_are_independent(creds_path: Path) -> None:
    creds.update(endpoint_keys={"alpha": "k1", "bravo": "k2"})
    assert creds.endpoint_keys["alpha"] == "k1"
    assert creds.endpoint_keys["bravo"] == "k2"


# ── Cross-slot isolation ────────────────────────────────────────────


def test_update_preserves_other_slots(creds_path: Path) -> None:
    creds.update(
        github_token="ghp_tok",
        openai_api_key="sk-key",
        claude=OAuthCreds(access_token="claude"),
        endpoint_keys={"ep": "ep-key"},
    )

    # Overwrite one field
    creds.update(openai_api_key="sk-new")

    assert creds.github_token == "ghp_tok"
    assert creds.openai_api_key == "sk-new"
    assert creds.claude.access_token == "claude"
    assert creds.endpoint_keys["ep"] == "ep-key"
