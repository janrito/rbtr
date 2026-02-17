"""Tests for rbtr.providers — pydantic-ai model construction from stored credentials."""

import pytest

from rbtr import RbtrError
from rbtr.config import config
from rbtr.creds import OAuthCreds

# ── Anthropic ────────────────────────────────────────────────────────


def _claude_oauth(**overrides: object) -> OAuthCreds:
    defaults: dict = {
        "access_token": "test-bearer-token",
        "refresh_token": "ref",
        "expires_at": 9999999999,
    }
    return OAuthCreds(**{**defaults, **overrides})


def test_build_claude_model(mocker) -> None:
    mocker.patch(
        "rbtr.providers.claude.ensure_credentials",
        return_value=_claude_oauth(),
    )
    from rbtr.providers.claude import build_model

    model = build_model()

    from pydantic_ai.models.anthropic import AnthropicModel

    assert isinstance(model, AnthropicModel)
    assert model.model_name == config.providers.claude.default_model
    assert model.client.auth_token == "test-bearer-token"


def test_build_claude_model_sets_oauth_headers(mocker) -> None:
    """OAuth bearer auth requires Claude Code identity headers."""
    mocker.patch(
        "rbtr.providers.claude.ensure_credentials",
        return_value=_claude_oauth(),
    )
    from rbtr.providers.claude import build_model

    model = build_model()

    headers = model.client.default_headers
    assert headers["anthropic-beta"] == config.providers.claude.oauth_beta
    assert headers["user-agent"] == config.providers.claude.oauth_user_agent
    assert headers["x-app"] == "cli"


def test_build_claude_model_custom_name(mocker) -> None:
    mocker.patch(
        "rbtr.providers.claude.ensure_credentials",
        return_value=_claude_oauth(),
    )
    from rbtr.providers.claude import build_model

    model = build_model("claude-opus-4-20250514")
    assert model.model_name == "claude-opus-4-20250514"


def test_build_claude_model_no_credentials(mocker) -> None:
    mocker.patch(
        "rbtr.providers.claude.ensure_credentials",
        side_effect=RbtrError("Not connected"),
    )
    from rbtr.providers.claude import build_model

    with pytest.raises(RbtrError, match="Not connected"):
        build_model()


# ── OpenAI ───────────────────────────────────────────────────────────


def test_build_openai_model(creds_path) -> None:
    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test-key-123")
    from rbtr.providers.openai import build_model

    model = build_model()

    from pydantic_ai.models.openai import OpenAIResponsesModel

    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == config.providers.openai.default_model
    assert model.client.api_key == "sk-test-key-123"


def test_build_openai_model_custom_name(creds_path) -> None:
    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")
    from rbtr.providers.openai import build_model

    model = build_model("gpt-4o-mini")
    assert model.model_name == "gpt-4o-mini"


def test_build_openai_model_no_key(creds_path) -> None:
    from rbtr.providers.openai import build_model

    with pytest.raises(RbtrError, match="No OpenAI API key"):
        build_model()


# ── ChatGPT (Codex) ──────────────────────────────────────────────────


def test_build_chatgpt_model(mocker) -> None:
    mocker.patch(
        "rbtr.providers.openai_codex.ensure_credentials",
        return_value=OAuthCreds(
            access_token="jwt-token",
            refresh_token="ref",
            expires_at=9999999999,
            account_id="acct_123",
        ),
    )
    from rbtr.providers.openai_codex import build_model

    model = build_model()

    from pydantic_ai.models.openai import OpenAIResponsesModel

    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == config.providers.chatgpt.default_model
    assert model.client.api_key == "jwt-token"
    assert str(model.client.base_url).rstrip("/") == config.providers.chatgpt.codex_base_url
    assert model.client.default_headers["chatgpt-account-id"] == "acct_123"
    assert model.client.default_headers["originator"] == "rbtr"


# ── build_model dispatch ─────────────────────────────────────────────


def test_build_model_prefers_claude(creds_path, mocker) -> None:
    """When multiple providers are configured, Anthropic wins."""
    from rbtr.creds import creds

    creds.update(claude=_claude_oauth(), openai_api_key="sk-key")
    mock_build = mocker.patch("rbtr.providers.claude.build_model")
    from rbtr.providers import build_model

    build_model()

    mock_build.assert_called_once()


def test_build_model_falls_back_to_chatgpt(creds_path, mocker) -> None:
    from rbtr.creds import creds

    creds.update(chatgpt=OAuthCreds(access_token="t", refresh_token="r", expires_at=9e9))
    mock_build = mocker.patch("rbtr.providers.openai_codex.build_model")
    from rbtr.providers import build_model

    build_model()

    mock_build.assert_called_once()


def test_build_model_falls_back_to_openai(creds_path, mocker) -> None:
    from rbtr.creds import creds

    creds.update(openai_api_key="sk-key")
    mock_build = mocker.patch("rbtr.providers.openai.build_model")
    from rbtr.providers import build_model

    build_model()

    mock_build.assert_called_once()


def test_build_model_no_provider(creds_path) -> None:
    from rbtr.providers import build_model

    with pytest.raises(RbtrError, match="No LLM connected"):
        build_model()


# ── build_model by name ──────────────────────────────────────────────


def test_build_model_by_name_claude(creds_path, mocker) -> None:
    from rbtr.creds import creds

    creds.update(claude=_claude_oauth())
    mock_build = mocker.patch("rbtr.providers.claude.build_model")
    from rbtr.providers import build_model

    build_model("claude/claude-opus-4-20250514")
    mock_build.assert_called_once_with("claude-opus-4-20250514")


def test_build_model_by_name_unknown_raises(mocker) -> None:
    mocker.patch("rbtr.providers.endpoint.load_endpoint", return_value=None)
    from rbtr.providers import build_model

    with pytest.raises(RbtrError, match="Unknown provider"):
        build_model("fakeprovider/some-model")


def test_build_model_by_name_no_slash_raises() -> None:
    from rbtr.providers import build_model

    with pytest.raises(RbtrError, match="Invalid model format"):
        build_model("noslash")
