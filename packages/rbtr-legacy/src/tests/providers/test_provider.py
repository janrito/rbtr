"""Tests for rbtr.providers — pydantic-ai model construction from stored credentials.

Uses shared credential data so every test verifies behaviour against
named, inspectable token sets rather than anonymous inline strings.
"""

from pathlib import Path

import pytest
from anthropic import AsyncAnthropic
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pytest_mock import MockerFixture

from rbtr.creds import OAuthCreds, creds
from rbtr.exceptions import RbtrError
from rbtr.providers import (
    PROVIDERS,
    BuiltinProvider,
    Provider,
    build_model,
    claude,
    fireworks,
    openai,
    openai_codex,
    openrouter,
)
from rbtr.providers.claude import provider as claude_prov
from rbtr.providers.endpoint import Endpoint
from rbtr.providers.fireworks import provider as fireworks_prov
from rbtr.providers.openai import provider as openai_prov
from rbtr.providers.openai_codex import provider as codex_prov
from rbtr.providers.openrouter import provider as openrouter_prov

# ── Shared test data ─────────────────────────────────────────────────

_CLAUDE_OAUTH = OAuthCreds(
    access_token="test-bearer-token",
    refresh_token="ref",
    expires_at=9999999999,
)

_CHATGPT_OAUTH = OAuthCreds(
    access_token="jwt-token",
    refresh_token="ref",
    expires_at=9999999999,
    account_id="acct_123",
)

_OPENAI_KEY = "sk-test-key-123"
_FIREWORKS_KEY = "fw-test-key-456"
_OPENROUTER_KEY = "sk-or-v1-test-key-789"


# ── Anthropic ────────────────────────────────────────────────────────


def test_build_claude_model(mocker: MockerFixture) -> None:
    mocker.patch("rbtr.providers.claude.ensure_credentials", return_value=_CLAUDE_OAUTH)

    model = claude_prov.build_model("claude-sonnet-4-20250514")

    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-sonnet-4-20250514"

    assert isinstance(model.client, AsyncAnthropic)
    assert model.client.auth_token == _CLAUDE_OAUTH.access_token


def test_build_claude_model_sets_oauth_headers(mocker: MockerFixture) -> None:
    """OAuth bearer auth requires Claude Code identity headers."""
    mocker.patch("rbtr.providers.claude.ensure_credentials", return_value=_CLAUDE_OAUTH)

    model = claude_prov.build_model("claude-sonnet-4-20250514")

    headers = model.client.default_headers  # type: ignore[attr-defined]  # mock attr on autospecced AnthropicModel
    assert headers["anthropic-beta"] == claude._OAUTH_BETA
    assert headers["user-agent"] == claude._OAUTH_USER_AGENT
    assert headers["x-app"] == "cli"


def test_build_claude_model_custom_name(mocker: MockerFixture) -> None:
    mocker.patch("rbtr.providers.claude.ensure_credentials", return_value=_CLAUDE_OAUTH)

    model = claude_prov.build_model("claude-opus-4-20250514")
    assert model.model_name == "claude-opus-4-20250514"


def test_build_claude_model_no_credentials(mocker: MockerFixture) -> None:
    mocker.patch(
        "rbtr.providers.claude.ensure_credentials",
        side_effect=RbtrError("Not connected"),
    )

    with pytest.raises(RbtrError, match="Not connected"):
        claude_prov.build_model("claude-sonnet-4-20250514")


# ── OpenAI ───────────────────────────────────────────────────────────


def test_build_openai_model(creds_path: Path) -> None:

    creds.update(openai_api_key=_OPENAI_KEY)

    model = openai_prov.build_model("gpt-4o")

    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == "gpt-4o"
    assert model.client.api_key == _OPENAI_KEY


def test_build_openai_model_custom_name(creds_path: Path) -> None:

    creds.update(openai_api_key=_OPENAI_KEY)

    model = openai_prov.build_model("gpt-4o-mini")
    assert model.model_name == "gpt-4o-mini"


def test_build_openai_model_no_key(creds_path: Path) -> None:

    with pytest.raises(RbtrError, match="No OpenAI API key"):
        openai_prov.build_model("gpt-4o")


# ── ChatGPT (Codex) ──────────────────────────────────────────────────


def test_build_chatgpt_model(mocker: MockerFixture) -> None:
    mocker.patch(
        "rbtr.providers.openai_codex.ensure_credentials",
        return_value=_CHATGPT_OAUTH,
    )

    model = codex_prov.build_model("gpt-4o")

    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == "gpt-4o"
    assert model.client.api_key == _CHATGPT_OAUTH.access_token
    assert str(model.client.base_url).rstrip("/") == openai_codex._CODEX_BASE_URL
    assert model.client.default_headers["chatgpt-account-id"] == _CHATGPT_OAUTH.account_id
    assert model.client.default_headers["originator"] == "rbtr"


# ── Fireworks ─────────────────────────────────────────────────────────


def test_build_fireworks_model(creds_path: Path) -> None:

    creds.update(fireworks_api_key=_FIREWORKS_KEY)

    model = fireworks_prov.build_model("accounts/fireworks/models/llama-v3p1-70b-instruct")

    assert isinstance(model, OpenAIChatModel)
    assert model._provider.name == "fireworks"


def test_build_fireworks_model_custom_name(creds_path: Path) -> None:

    creds.update(fireworks_api_key=_FIREWORKS_KEY)

    model = fireworks_prov.build_model("accounts/fireworks/models/kimi-k2p5")
    assert model.model_name == "accounts/fireworks/models/kimi-k2p5"


def test_build_fireworks_model_no_key(creds_path: Path) -> None:

    with pytest.raises(RbtrError, match="No Fireworks API key"):
        fireworks_prov.build_model("accounts/fireworks/models/llama-v3p1-70b-instruct")


# ── OpenRouter ────────────────────────────────────────────────────────


def test_build_openrouter_model(creds_path: Path) -> None:

    creds.update(openrouter_api_key=_OPENROUTER_KEY)

    model = openrouter_prov.build_model("anthropic/claude-sonnet-4-20250514")

    assert isinstance(model, OpenRouterModel)
    assert model._provider.name == "openrouter"


def test_build_openrouter_model_custom_name(creds_path: Path) -> None:

    creds.update(openrouter_api_key=_OPENROUTER_KEY)

    model = openrouter_prov.build_model("anthropic/claude-sonnet-4-20250514")
    assert model.model_name == "anthropic/claude-sonnet-4-20250514"


def test_build_openrouter_model_no_key(creds_path: Path) -> None:

    with pytest.raises(RbtrError, match="No OpenRouter API key"):
        openrouter_prov.build_model("anthropic/claude-sonnet-4-20250514")


# ── build_model by name ──────────────────────────────────────────────


def test_build_model_by_name_claude(creds_path: Path, mocker: MockerFixture) -> None:

    creds.update(claude=_CLAUDE_OAUTH)
    mock_build = mocker.patch.object(claude.provider, "build_model")

    build_model("claude/claude-opus-4-20250514")
    mock_build.assert_called_once_with("claude-opus-4-20250514")


def test_build_model_by_name_fireworks(creds_path: Path, mocker: MockerFixture) -> None:

    creds.update(fireworks_api_key=_FIREWORKS_KEY)
    mock_build = mocker.patch.object(fireworks.provider, "build_model")

    build_model("fireworks/accounts/fireworks/models/kimi-k2p5")
    mock_build.assert_called_once_with("accounts/fireworks/models/kimi-k2p5")


def test_build_model_by_name_openrouter(creds_path: Path, mocker: MockerFixture) -> None:

    creds.update(openrouter_api_key=_OPENROUTER_KEY)
    mock_build = mocker.patch.object(openrouter.provider, "build_model")

    build_model("openrouter/anthropic/claude-sonnet-4-20250514")
    mock_build.assert_called_once_with("anthropic/claude-sonnet-4-20250514")


def test_build_model_by_name_unknown_raises(mocker: MockerFixture) -> None:
    mocker.patch("rbtr.providers.endpoint.load_endpoint", return_value=None)

    with pytest.raises(RbtrError, match="Unknown provider"):
        build_model("fakeprovider/some-model")


def test_build_model_by_name_chatgpt(creds_path: Path, mocker: MockerFixture) -> None:

    creds.update(chatgpt=_CHATGPT_OAUTH)
    mock_build = mocker.patch.object(openai_codex.provider, "build_model")

    build_model("chatgpt/gpt-4o")
    mock_build.assert_called_once_with("gpt-4o")


def test_build_model_by_name_openai(creds_path: Path, mocker: MockerFixture) -> None:

    creds.update(openai_api_key=_OPENAI_KEY)
    mock_build = mocker.patch.object(openai.provider, "build_model")

    build_model("openai/gpt-4o-mini")
    mock_build.assert_called_once_with("gpt-4o-mini")


def test_build_model_by_name_chatgpt_not_connected(creds_path: Path) -> None:

    with pytest.raises(RbtrError, match="Not connected"):
        build_model("chatgpt/gpt-4o")


def test_build_model_by_name_openai_not_connected(creds_path: Path) -> None:

    with pytest.raises(RbtrError, match="No OpenAI API key"):
        build_model("openai/gpt-4o")


def test_build_model_by_name_claude_not_connected(creds_path: Path) -> None:

    with pytest.raises(RbtrError, match=r"Not connected|No .* key"):
        build_model("claude/claude-sonnet-4-20250514")


def test_build_model_by_name_endpoint(
    creds_path: Path, config_path: Path, mocker: MockerFixture
) -> None:

    mocker.patch(
        "rbtr.providers.endpoint.load_endpoint",
        return_value=Endpoint(name="myep", base_url="http://localhost:11434/v1", api_key=""),
    )
    mock_build = mocker.patch("rbtr.providers.endpoint.build_model")
    build_model("myep/llama3")
    mock_build.assert_called_once_with("myep", "llama3")


def test_build_model_by_name_no_slash_raises() -> None:

    with pytest.raises(RbtrError, match="Invalid model format"):
        build_model("noslash")


# ── Provider contract conformance ────────────────────────────────────


def test_all_builtin_providers_registered() -> None:
    """Every BuiltinProvider member has an entry in PROVIDERS."""

    for provider in BuiltinProvider:
        assert provider in PROVIDERS, (
            f"{provider} is not in PROVIDERS — add it when wiring a new provider"
        )


def test_all_providers_satisfy_protocol() -> None:
    """Every module in PROVIDERS satisfies the Provider protocol."""

    for provider, mod in PROVIDERS.items():
        assert isinstance(mod, Provider), f"{provider} module does not satisfy Provider protocol"
        assert mod.GENAI_ID, f"{provider}.GENAI_ID is empty"
        assert mod.LABEL, f"{provider}.LABEL is empty"
