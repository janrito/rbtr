"""Tests for engine/model_settings.py — model settings resolution."""

from __future__ import annotations

from pytest_mock import MockerFixture

from rbtr.llm.model_settings import resolve_model_settings


def test_resolve_model_settings_skips_effort_when_unsupported(mocker: MockerFixture) -> None:
    """When effort_supported=False, effort settings are omitted."""
    from pydantic_ai.models.anthropic import AnthropicModel

    mock_model = mocker.MagicMock(spec=AnthropicModel)

    settings = resolve_model_settings(mock_model, "claude/claude-sonnet-4-5-20250929")
    # Default effort is MEDIUM → should produce settings
    assert settings is not None

    # With effort_supported=False → no effort settings
    settings_no = resolve_model_settings(
        mock_model, "claude/claude-sonnet-4-5-20250929", effort_supported=False
    )
    assert settings_no is None
