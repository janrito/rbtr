"""API error classification heuristics.

Pure functions that inspect `ModelHTTPError` to decide whether the
error is a context overflow, unsupported-effort rejection, etc.
"""

from __future__ import annotations

from http import HTTPStatus

from pydantic_ai.exceptions import ModelHTTPError

# Keywords in API error messages that indicate a context-length issue.
_OVERFLOW_KEYWORDS = (
    "context length",
    "context window",
    "maximum context",
    "token limit",
    "too many tokens",
    "too long",
    "max_tokens",
    "prompt is too long",
    "input is too long",
    "request too large",
    "content_too_large",
)

# Keywords paired with "effort" that signal the parameter was rejected.
_EFFORT_REJECTION_KEYWORDS = (
    "not support",
    "unsupported",
    "not available",
    "not allowed",
    "unknown parameter",
    "invalid parameter",
    "unrecognized",
)


def is_effort_unsupported(exc: ModelHTTPError) -> bool:
    """Does the API error indicate that the model doesn't support the effort parameter?"""
    msg = str(exc).lower()
    return "effort" in msg and any(kw in msg for kw in _EFFORT_REJECTION_KEYWORDS)


def is_context_overflow(exc: ModelHTTPError) -> bool:
    """Heuristic: does the API error indicate a context-length problem?"""
    if exc.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
        return True
    if exc.status_code == HTTPStatus.BAD_REQUEST:
        msg = str(exc).lower()
        return any(kw in msg for kw in _OVERFLOW_KEYWORDS)
    return False
