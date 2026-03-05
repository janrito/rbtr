"""Scrub secrets from free-text strings.

Applies regex patterns for common API key formats before
persisting diagnostics (tracebacks, error bodies) to the
local SQLite store.  Not a general-purpose PII redactor —
targets the key formats used by LLM providers (OpenAI,
Anthropic, Bearer tokens, key=value pairs).
"""

from __future__ import annotations

import re

# Patterns that match common API key formats.  Each pattern
# preserves the prefix/label so the key type is identifiable.
_SCRUB_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Bearer tokens in headers: "Bearer sk-abc123..." → "Bearer [REDACTED]"
    (re.compile(r"(?i)(bearer\s+)\S+"), r"\1[REDACTED]"),
    # Anthropic: sk-ant-...
    (re.compile(r"(sk-ant-)\S+"), r"\1[REDACTED]"),
    # OpenAI: sk-... (but not sk-ant-)
    (re.compile(r"(sk-(?!ant-))\S+"), r"\1[REDACTED]"),
    # Fireworks: fw-...
    (re.compile(r"(fw-)\S+"), r"\1[REDACTED]"),
    # Generic key=value in URLs/headers: api_key=abc123
    (
        re.compile(r"(?i)((?:api[_-]?key|token|secret|password|authorization)\s*[=:]\s*)\S+"),
        r"\1[REDACTED]",
    ),
]


def scrub_secrets(text: str) -> str:
    """Remove API keys and tokens from *text*.

    Applies regex patterns for common key formats (OpenAI,
    Anthropic, Bearer tokens, key=value pairs).  Preserves
    the key prefix so the type is identifiable in diagnostics.
    """
    for pattern, replacement in _SCRUB_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
