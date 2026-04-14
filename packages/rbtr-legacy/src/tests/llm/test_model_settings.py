"""Tests for model_settings dispatch — effort skipped for NONE."""

from __future__ import annotations

from pydantic_ai.models.anthropic import AnthropicModel
from pytest_mock import MockerFixture

from rbtr_legacy.config import ThinkingEffort
from rbtr_legacy.providers import model_settings


def test_model_settings_returns_effort(mocker: MockerFixture) -> None:
    """A known provider + non-NONE effort → settings returned."""

    mock_model = mocker.MagicMock(spec=AnthropicModel)
    settings = model_settings(
        "claude/claude-sonnet-4-5-20250929", mock_model, ThinkingEffort.MEDIUM
    )
    assert settings is not None


def test_model_settings_none_for_none_effort(mocker: MockerFixture) -> None:
    """NONE effort → provider returns None (no effort applied)."""

    mock_model = mocker.MagicMock(spec=AnthropicModel)
    settings = model_settings("claude/claude-sonnet-4-5-20250929", mock_model, ThinkingEffort.NONE)
    assert settings is None
