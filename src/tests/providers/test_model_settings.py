"""Tests for build_model_settings — provider-specific effort mapping."""

from __future__ import annotations

import pytest

from rbtr.config import ThinkingEffort
from rbtr.providers import build_model_settings

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
    from rbtr.providers.claude import build_model

    return build_model()


def _openai_model(creds_path):
    """Build a real OpenAI model with a test API key."""
    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test-key")
    from rbtr.providers.openai import build_model

    return build_model()


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
    model = _claude_model(mocker)
    settings = build_model_settings(model, effort)
    assert settings is not None
    assert settings.get("anthropic_effort") == expected_value


def test_anthropic_none_returns_none(mocker) -> None:
    """NONE effort → no settings (caller should not pass model_settings)."""
    model = _claude_model(mocker)
    settings = build_model_settings(model, ThinkingEffort.NONE)
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
    model = _openai_model(creds_path)
    settings = build_model_settings(model, effort)
    assert settings is not None
    assert settings.get("openai_reasoning_effort") == expected_value


def test_openai_none_returns_none(creds_path) -> None:
    model = _openai_model(creds_path)
    settings = build_model_settings(model, ThinkingEffort.NONE)
    assert settings is None


# ── Unsupported model ────────────────────────────────────────────────


def test_unsupported_model_returns_none() -> None:
    """A model type we don't recognise → None for all effort levels."""
    from unittest.mock import MagicMock

    fake_model = MagicMock()
    for effort in ThinkingEffort:
        assert build_model_settings(fake_model, effort) is None
