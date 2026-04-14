"""Tests for `rbtr.scrub` — secret scrubbing."""

from __future__ import annotations

import pytest

from rbtr_legacy.sessions.scrub import scrub_secrets


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # OpenAI key
        (
            "Error calling https://api.openai.com with sk-proj-abc123def456ghi789",
            "Error calling https://api.openai.com with sk-[REDACTED]",
        ),
        # Anthropic key
        (
            "Authorization failed: sk-ant-api03-long-key-here",
            "Authorization failed: sk-ant-[REDACTED]",
        ),
        # Bearer token
        (
            "Header: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
            "Header: Bearer [REDACTED]",
        ),
        # key=value in URL
        (
            "GET https://api.example.com/v1?api_key=secret123&format=json",
            "GET https://api.example.com/v1?api_key=[REDACTED]",
        ),
        # Mixed case header colon-separated
        (
            "API-Key: xai-abcdef123456",
            "API-Key: [REDACTED]",
        ),
        # Fireworks key
        (
            "Error from Fireworks API with fw-abc123def456ghi789jkl",
            "Error from Fireworks API with fw-[REDACTED]",
        ),
        # No secrets — unchanged
        (
            "TypeError: 'NoneType' object is not subscriptable",
            "TypeError: 'NoneType' object is not subscriptable",
        ),
        # Multiple keys in one string
        (
            "keys: sk-proj-abc123 and sk-ant-api03-xyz789",
            "keys: sk-[REDACTED] and sk-ant-[REDACTED]",
        ),
        # password=value
        (
            "connection string: password=hunter2 host=localhost",
            "connection string: password=[REDACTED] host=localhost",
        ),
    ],
    ids=[
        "openai_key",
        "anthropic_key",
        "bearer_token",
        "url_api_key",
        "mixed_case_header",
        "fireworks_key",
        "no_secrets",
        "multiple_keys",
        "password",
    ],
)
def test_scrub_secrets(raw: str, expected: str) -> None:
    """`scrub_secrets` redacts API keys while preserving context."""
    assert scrub_secrets(raw) == expected
