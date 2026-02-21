"""History repair helpers for cross-provider compatibility."""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ThinkingPart


def is_history_format_error(exc: Exception) -> bool:
    """Check if an API error is caused by malformed history items.

    PydanticAI stores provider-specific reasoning IDs in message history.
    When history is replayed to a different provider (or API variant),
    these IDs may be rejected (e.g. OpenAI Responses API expects ``rs_*``).
    """
    msg = str(exc).lower()
    # "Invalid 'input[6].id': 'reasoning_content'" — wrong ID format
    if "invalid" in msg and ("id" in msg or "input" in msg):
        return True
    # "Item 'fc_...' was provided without its required 'reasoning' item"
    return "provided without" in msg and "reasoning" in msg


def demote_thinking(history: list[ModelMessage]) -> list[ModelMessage]:
    """Return history with ThinkingParts converted to plain TextParts.

    Wraps thinking content in ``<thinking>`` tags so the model can
    still see prior reasoning, without the provider-specific IDs
    that cause cross-provider errors.
    """
    cleaned: list[ModelMessage] = []
    for msg in history:
        if isinstance(msg, ModelResponse):
            parts = [
                TextPart(content=f"<thinking>\n{p.content}\n</thinking>")
                if isinstance(p, ThinkingPart) and p.content
                else p
                for p in msg.parts
                if not isinstance(p, ThinkingPart) or p.content
            ]
            if parts:
                cleaned.append(replace(msg, parts=parts))
        else:
            cleaned.append(msg)
    return cleaned
