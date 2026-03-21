"""Shared fixtures and assertions for session tests."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)


def assert_ordering(messages: list[ModelMessage]) -> None:
    """Request before response, no consecutive responses."""
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelResponse) and i == 0:
            pytest.fail("message 0 is a ModelResponse — history starts with assistant")
        if isinstance(msg, ModelResponse) and isinstance(messages[i - 1], ModelResponse):
            pytest.fail(f"messages {i - 1} and {i} are consecutive ModelResponses")


def assert_tool_pairing(messages: list[ModelMessage]) -> None:
    """Every tool call has a matching return in the next request."""
    for i, msg in enumerate(messages):
        if not isinstance(msg, ModelResponse):
            continue
        call_ids = {p.tool_call_id for p in msg.parts if isinstance(p, ToolCallPart)}
        if not call_ids:
            continue
        assert i + 1 < len(messages), f"response {i} has tool calls but no next message"
        next_msg = messages[i + 1]
        assert isinstance(next_msg, ModelRequest), (
            f"message {i + 1} after tool calls is not a request"
        )
        return_ids = {
            p.tool_call_id
            for p in next_msg.parts
            if isinstance(p, (ToolReturnPart, RetryPromptPart))
        }
        assert call_ids == return_ids, (
            f"response {i}: call IDs {call_ids} != return IDs {return_ids}"
        )


def assert_messages_match(
    expected: list[ModelMessage],
    actual: list[ModelMessage],
) -> None:
    """Message-by-message, part-by-part comparison.

    Checks type, part count, and field values for all part types.
    """
    assert len(actual) >= len(expected), (
        f"expected at least {len(expected)} messages, got {len(actual)}"
    )
    for i, (exp, act) in enumerate(zip(expected, actual, strict=False)):
        assert type(act) is type(exp), (
            f"message {i}: expected {type(exp).__name__}, got {type(act).__name__}"
        )
        assert len(act.parts) == len(exp.parts), (
            f"message {i}: expected {len(exp.parts)} parts, got {len(act.parts)}"
        )
        for j, (ep, ap) in enumerate(zip(exp.parts, act.parts, strict=True)):
            label = f"message {i} part {j}"
            if isinstance(ep, TextPart) and isinstance(ap, TextPart):
                assert ap.content == ep.content, f"{label}: text content"
            elif isinstance(ep, ThinkingPart) and isinstance(ap, ThinkingPart):
                assert ap.content == ep.content, f"{label}: thinking content"
                assert ap.signature == ep.signature, f"{label}: signature"
                assert ap.provider_name == ep.provider_name, f"{label}: provider"
                assert ap.id == ep.id, f"{label}: id"
            elif isinstance(ep, ToolCallPart) and isinstance(ap, ToolCallPart):
                assert ap.tool_name == ep.tool_name, f"{label}: tool name"
                assert ap.tool_call_id == ep.tool_call_id, f"{label}: call id"
            elif isinstance(ep, ToolReturnPart) and isinstance(ap, ToolReturnPart):
                assert ap.tool_name == ep.tool_name, f"{label}: tool name"
                assert ap.tool_call_id == ep.tool_call_id, f"{label}: call id"
                assert ap.content == ep.content, f"{label}: content"
            else:
                assert type(ap) is type(ep), f"{label}: type mismatch"
