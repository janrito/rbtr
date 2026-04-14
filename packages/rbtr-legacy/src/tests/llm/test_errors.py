"""Tests for engine/errors.py — API error classification heuristics."""

from __future__ import annotations

from http import HTTPStatus

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from rbtr_legacy.llm.errors import is_context_overflow, is_effort_unsupported


def _make_http_error(status: HTTPStatus, body: str) -> ModelHTTPError:
    """Construct a ModelHTTPError with given status and body text."""
    return ModelHTTPError(status, "test-model", body)


# ── is_context_overflow ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (HTTPStatus.BAD_REQUEST, "This model's maximum context length is 128000 tokens"),
        (HTTPStatus.BAD_REQUEST, "prompt is too long: 150000 tokens > 128000 token limit"),
        (HTTPStatus.BAD_REQUEST, "Request too large for model"),
        (HTTPStatus.BAD_REQUEST, "input is too long (150000 tokens, max 128000)"),
        (HTTPStatus.BAD_REQUEST, "content_too_large: message exceeds context window"),
        (HTTPStatus.BAD_REQUEST, "too many tokens in the prompt"),
        (HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Payload too large"),
    ],
)
def test_is_context_overflow_positive(status: HTTPStatus, body: str) -> None:
    """Errors that indicate context overflow are detected."""
    exc = _make_http_error(status, body)
    assert is_context_overflow(exc)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (HTTPStatus.BAD_REQUEST, "Invalid API key"),
        (HTTPStatus.BAD_REQUEST, "malformed request body"),
        (HTTPStatus.UNAUTHORIZED, "Unauthorized"),
        (HTTPStatus.TOO_MANY_REQUESTS, "Rate limit exceeded"),
        (HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error"),
    ],
)
def test_is_context_overflow_negative(status: HTTPStatus, body: str) -> None:
    """Non-context errors are not misidentified."""
    exc = _make_http_error(status, body)
    assert not is_context_overflow(exc)


# ── is_effort_unsupported ────────────────────────────────────────────


@pytest.mark.parametrize(
    "body",
    [
        "This model does not support the effort parameter.",
        "Effort is not supported for this model",
        "unsupported parameter: effort",
        "effort parameter is not available for this model",
        "effort is not allowed with this configuration",
        "unknown parameter: effort",
        "invalid parameter 'effort'",
        "unrecognized parameter: effort",
    ],
)
def test_is_effort_unsupported_positive(body: str) -> None:
    """Error messages about unsupported effort are detected."""
    exc = _make_http_error(HTTPStatus.BAD_REQUEST, body)
    assert is_effort_unsupported(exc)


@pytest.mark.parametrize(
    "body",
    [
        "Invalid API key",
        "maximum context length exceeded",
        "malformed request body",
        "unsupported parameter: temperature",
        "not supported: streaming mode",
    ],
)
def test_is_effort_unsupported_negative(body: str) -> None:
    """Unrelated errors are not misidentified as effort-unsupported."""
    exc = _make_http_error(HTTPStatus.BAD_REQUEST, body)
    assert not is_effort_unsupported(exc)
