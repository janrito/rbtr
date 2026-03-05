"""Tests for provider model_settings — provider-specific effort mapping."""

from __future__ import annotations

import pytest

from rbtr.config import ThinkingEffort
from rbtr.providers import model_context_window
from rbtr.providers.endpoint import EndpointProvider, ModelMetadata

# ── Helpers ──────────────────────────────────────────────────────────


def _claude_model(mocker):
    """Build a real AnthropicModel with mocked credentials."""
    mocker.patch(
        "rbtr.providers.claude.ensure_credentials",
        return_value=mocker.MagicMock(
            access_token="test-token",
            refresh_token="ref",
            expires_at=9999999999,
        ),
    )
    from rbtr.providers.claude import provider as claude_prov

    return claude_prov.build_model("claude-sonnet-4-20250514")


def _openai_model(creds_path):
    """Build a real OpenAI model with a test API key."""
    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test-key")
    from rbtr.providers.openai import provider as openai_prov

    return openai_prov.build_model("gpt-4o")


# ── Anthropic effort mapping ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("effort", "expected_value"),
    [
        (ThinkingEffort.LOW, "low"),
        (ThinkingEffort.MEDIUM, "medium"),
        (ThinkingEffort.HIGH, "high"),
        (ThinkingEffort.MAX, "max"),
    ],
)
def test_anthropic_effort_levels(mocker, effort, expected_value) -> None:
    """Each effort level maps to the correct anthropic_effort value."""
    from rbtr.providers.claude import provider as claude_prov

    model = _claude_model(mocker)
    settings = claude_prov.model_settings("claude-sonnet-4-20250514", model, effort)
    assert settings is not None
    assert settings.get("anthropic_effort") == expected_value


def test_anthropic_none_returns_none(mocker) -> None:
    """NONE effort → no settings (caller should not pass model_settings)."""
    from rbtr.providers.claude import provider as claude_prov

    model = _claude_model(mocker)
    settings = claude_prov.model_settings("claude-sonnet-4-20250514", model, ThinkingEffort.NONE)
    assert settings is None


# ── OpenAI effort mapping ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("effort", "expected_value"),
    [
        (ThinkingEffort.LOW, "low"),
        (ThinkingEffort.MEDIUM, "medium"),
        (ThinkingEffort.HIGH, "high"),
        (ThinkingEffort.MAX, "xhigh"),
    ],
)
def test_openai_effort_levels(creds_path, effort, expected_value) -> None:
    """Each effort level maps to the correct openai_reasoning_effort value."""
    from rbtr.providers.openai import provider as openai_prov

    model = _openai_model(creds_path)
    settings = openai_prov.model_settings("gpt-4o", model, effort)
    assert settings is not None
    assert settings.get("openai_reasoning_effort") == expected_value


def test_openai_none_returns_none(creds_path) -> None:
    from rbtr.providers.openai import provider as openai_prov

    model = _openai_model(creds_path)
    settings = openai_prov.model_settings("gpt-4o", model, ThinkingEffort.NONE)
    assert settings is None


# ── Unsupported model ────────────────────────────────────────────────


def test_unsupported_model_returns_none() -> None:
    """A model type we don't recognise → None for all effort levels."""
    from unittest.mock import MagicMock

    from rbtr.providers.google import provider as google_prov

    fake_model = MagicMock()
    for effort in ThinkingEffort:
        assert google_prov.model_settings("gemini-2.5-pro", fake_model, effort) is None


# ── Endpoint model settings (auto-fetched metadata) ──────────────────


def test_endpoint_model_settings_from_metadata(mocker) -> None:
    """Auto-fetched context_window → max_tokens = context_window // 2."""
    from unittest.mock import MagicMock

    mocker.patch(
        "rbtr.providers.endpoint.fetch_model_metadata",
        return_value=ModelMetadata(context_window=196608),
    )
    prov = EndpointProvider("myep")
    settings = prov.model_settings("some-model", MagicMock(), ThinkingEffort.NONE)
    assert settings is not None
    assert settings["max_tokens"] == 98304  # 196608 // 2


def test_endpoint_model_settings_caps_at_131072(mocker) -> None:
    """For very large context windows, max_tokens is capped at 128 k."""
    from unittest.mock import MagicMock

    mocker.patch(
        "rbtr.providers.endpoint.fetch_model_metadata",
        return_value=ModelMetadata(context_window=1_000_000),
    )
    prov = EndpointProvider("myep")
    settings = prov.model_settings("big-model", MagicMock(), ThinkingEffort.NONE)
    assert settings is not None
    assert settings["max_tokens"] == 131_072


def test_endpoint_model_settings_none_when_no_metadata(mocker) -> None:
    from unittest.mock import MagicMock

    mocker.patch(
        "rbtr.providers.endpoint.fetch_model_metadata",
        return_value=None,
    )
    prov = EndpointProvider("myep")
    assert prov.model_settings("some-model", MagicMock(), ThinkingEffort.NONE) is None


# ── model_context_window (generic lookup) ─────────────────────────────


@pytest.mark.parametrize(
    ("model_name", "expected_window"),
    [
        ("claude/claude-sonnet-4-20250514", 200_000),
        ("openai/gpt-4o", 128_000),
        ("chatgpt/gpt-4o", 128_000),
    ],
)
def test_model_context_window_builtin(model_name: str, expected_window: int) -> None:
    """Built-in providers resolve context windows via genai-prices."""
    ctx = model_context_window(model_name)
    assert ctx == expected_window


def test_model_context_window_endpoint(mocker) -> None:
    """model_context_window prefers endpoint metadata over genai-prices."""
    from rbtr.providers.endpoint import Endpoint

    mocker.patch(
        "rbtr.providers.endpoint.load_endpoint",
        return_value=Endpoint(name="myep", base_url="http://localhost:11434/v1", api_key=""),
    )
    mocker.patch(
        "rbtr.providers.endpoint.fetch_model_metadata",
        return_value=ModelMetadata(context_window=327_680),
    )
    ctx = model_context_window("myep/custom-model")
    assert ctx == 327_680


def test_model_context_window_chatgpt_prefers_codex_metadata(mocker) -> None:
    """ChatGPT model metadata overrides missing/unknown genai-prices entries."""
    mocker.patch(
        "rbtr.providers.openai_codex.fetch_model_metadata",
        return_value=ModelMetadata(context_window=200_000),
    )
    ctx = model_context_window("chatgpt/o3-pro")
    assert ctx == 200_000


def test_model_context_window_chatgpt_falls_back_to_genai_prices(mocker) -> None:
    """When Codex metadata is unavailable, fallback to genai-prices."""
    mocker.patch(
        "rbtr.providers.openai_codex.fetch_model_metadata",
        return_value=None,
    )
    ctx = model_context_window("chatgpt/gpt-4o")
    assert ctx == 128_000


@pytest.mark.parametrize(
    "model_name",
    [None, "", "noslash"],
    ids=["none", "empty", "no-slash"],
)
def test_model_context_window_none_for_invalid(model_name: str | None) -> None:
    """Malformed or missing model names return None."""
    assert model_context_window(model_name) is None


def test_model_context_window_none_for_unknown_provider(mocker) -> None:
    mocker.patch("rbtr.providers.endpoint.load_endpoint", return_value=None)
    assert model_context_window("fakeprovider/unknown-model") is None


def test_model_context_window_none_for_unknown_model() -> None:
    """Unknown model under a known provider returns None."""
    assert model_context_window("claude/totally-fake-model-that-doesnt-exist") is None
