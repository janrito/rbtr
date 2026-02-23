"""Tests for session serialisation — prepare_row and roundtrips."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from rbtr.sessions.serialise import (
    MessageKind,
    SessionContext,
    deserialise_message,
    prepare_row,
    serialise_message,
)

# ── Shared context ───────────────────────────────────────────────────

CTX = SessionContext(
    session_id="ses-001",
    session_label="testowner/testrepo — main",
    repo_owner="testowner",
    repo_name="testrepo",
    model_name="claude/claude-sonnet-4-20250514",
)

# ── Shared test messages ─────────────────────────────────────────────

USER_REQUEST = ModelRequest(parts=[UserPromptPart(content="review src/tui.py")])

ASSISTANT_RESPONSE = ModelResponse(
    parts=[TextPart(content="I'll read the file.")],
    model_name="claude-sonnet-4-20250514",
    usage=RequestUsage(input_tokens=1200, output_tokens=50),
)

TOOL_CALL_RESPONSE = ModelResponse(
    parts=[
        TextPart(content="Let me check."),
        ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
        ToolCallPart(tool_name="grep", args={"pattern": "TODO"}, tool_call_id="tc2"),
    ],
    model_name="claude-sonnet-4-20250514",
    usage=RequestUsage(input_tokens=3500, output_tokens=120),
)

TOOL_RETURN_REQUEST = ModelRequest(
    parts=[
        ToolReturnPart(tool_name="read_file", content="def hello(): ...", tool_call_id="tc1"),
        ToolReturnPart(tool_name="grep", content="L42: # TODO fix", tool_call_id="tc2"),
    ]
)

THINKING_RESPONSE = ModelResponse(
    parts=[
        ThinkingPart(content="Let me reason about this..."),
        TextPart(content="Here's my analysis."),
    ],
    model_name="claude-sonnet-4-20250514",
    usage=RequestUsage(input_tokens=5000, output_tokens=200),
)


# ═══════════════════════════════════════════════════════════════════════
# serialise / deserialise roundtrip
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "message",
    [USER_REQUEST, ASSISTANT_RESPONSE, TOOL_CALL_RESPONSE, TOOL_RETURN_REQUEST, THINKING_RESPONSE],
    ids=["user", "assistant", "tool_call", "tool_return", "thinking"],
)
def test_roundtrip(message: ModelRequest | ModelResponse) -> None:
    """Serialise → deserialise produces an equal message."""
    json_str = serialise_message(message)
    restored = deserialise_message(json_str)
    assert restored == message


# ═══════════════════════════════════════════════════════════════════════
# prepare_row — ModelMessage routing
# ═══════════════════════════════════════════════════════════════════════


def test_user_request_row() -> None:
    """User prompt extracts user_text and kind='request'."""
    row = prepare_row(USER_REQUEST, context=CTX, row_id="r1")
    assert row.kind == MessageKind.REQUEST
    assert row.user_text == "review src/tui.py"
    assert row.tool_names is None
    assert row.input_tokens is None
    assert row.output_tokens is None
    assert row.message_json is not None
    assert row.session_id == "ses-001"
    assert row.model_name == "claude/claude-sonnet-4-20250514"


def test_assistant_response_row() -> None:
    """Plain assistant response has tokens, no tools."""
    row = prepare_row(ASSISTANT_RESPONSE, context=CTX, row_id="r2")
    assert row.kind == MessageKind.RESPONSE
    assert row.user_text is None
    assert row.tool_names is None
    assert row.input_tokens == 1200
    assert row.output_tokens == 50


def test_tool_call_response_row() -> None:
    """Response with tool calls extracts comma-separated tool_names."""
    row = prepare_row(TOOL_CALL_RESPONSE, context=CTX, row_id="r3")
    assert row.kind == MessageKind.RESPONSE
    assert row.tool_names == "read_file,grep"
    assert row.input_tokens == 3500
    assert row.output_tokens == 120


def test_tool_return_request_row() -> None:
    """Tool return request extracts tool_names from ToolReturnPart."""
    row = prepare_row(TOOL_RETURN_REQUEST, context=CTX, row_id="r4")
    assert row.kind == MessageKind.REQUEST
    assert row.tool_names == "read_file,grep"
    assert row.user_text is None


def test_thinking_response_row() -> None:
    """Response with thinking part still extracts tokens."""
    row = prepare_row(THINKING_RESPONSE, context=CTX, row_id="r5")
    assert row.kind == MessageKind.RESPONSE
    assert row.input_tokens == 5000
    assert row.output_tokens == 200
    assert row.tool_names is None


# ═══════════════════════════════════════════════════════════════════════
# prepare_row — str routing (command / shell)
# ═══════════════════════════════════════════════════════════════════════


def test_command_row() -> None:
    """/slash command produces kind='command', no JSON."""
    row = prepare_row("/review 42", context=CTX, row_id="r6", kind=MessageKind.COMMAND)
    assert row.kind == MessageKind.COMMAND
    assert row.user_text == "/review 42"
    assert row.message_json is None
    assert row.tool_names is None
    assert row.input_tokens is None
    assert row.cost is None


def test_shell_row() -> None:
    """!shell command produces kind='shell', no JSON."""
    row = prepare_row("!git log --oneline", context=CTX, row_id="r7", kind=MessageKind.SHELL)
    assert row.kind == MessageKind.SHELL
    assert row.user_text == "!git log --oneline"
    assert row.message_json is None


def test_str_without_kind_raises() -> None:
    """Passing a str without explicit kind raises ValueError."""
    with pytest.raises(ValueError, match="kind is required"):
        prepare_row("hello", context=CTX, row_id="r8")


def test_unsupported_type_raises() -> None:
    """Passing an unsupported type raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported message type"):
        prepare_row(42, context=CTX, row_id="r9")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════
# Context propagation
# ═══════════════════════════════════════════════════════════════════════


def test_context_propagated_to_row() -> None:
    """All SessionContext fields appear on the row."""
    ctx = SessionContext(
        session_id="ses-99",
        session_label="acme/app — pr-7",
        repo_owner="acme",
        repo_name="app",
        model_name="chatgpt/o3",
    )
    row = prepare_row(ASSISTANT_RESPONSE, context=ctx, row_id="r10", cost=0.0042)
    assert row.session_id == "ses-99"
    assert row.session_label == "acme/app — pr-7"
    assert row.repo_owner == "acme"
    assert row.repo_name == "app"
    assert row.model_name == "chatgpt/o3"
    assert row.cost == 0.0042


def test_cost_only_on_response() -> None:
    """Cost is passed explicitly and only applies to the row it's set on."""
    resp_row = prepare_row(ASSISTANT_RESPONSE, context=CTX, row_id="r10a", cost=0.05)
    assert resp_row.cost == 0.05
    req_row = prepare_row(USER_REQUEST, context=CTX, row_id="r10b")
    assert req_row.cost is None


def test_compacted_by_is_none() -> None:
    """Fresh rows always have compacted_by=None."""
    row = prepare_row(USER_REQUEST, context=CTX, row_id="r11")
    assert row.compacted_by is None


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════


def test_response_with_zero_usage() -> None:
    """Response with default (zero) usage produces 0 tokens."""
    msg = ModelResponse(parts=[TextPart(content="hi")], model_name="test")
    row = prepare_row(msg, context=CTX, row_id="r12")
    assert row.input_tokens == 0
    assert row.output_tokens == 0


def test_user_prompt_with_non_str_content() -> None:
    """UserPromptPart with list content is stringified."""
    msg = ModelRequest(parts=[UserPromptPart(content=["image", "data"])])  # type: ignore[arg-type]
    row = prepare_row(msg, context=CTX, row_id="r13")
    assert row.user_text is not None
    assert "image" in row.user_text


def test_mixed_request_parts() -> None:
    """Request with both UserPromptPart and ToolReturnPart."""
    msg = ModelRequest(
        parts=[
            UserPromptPart(content="explain this"),
            ToolReturnPart(tool_name="read_file", content="def foo(): ...", tool_call_id="tc1"),
        ]
    )
    row = prepare_row(msg, context=CTX, row_id="r14")
    assert row.user_text == "explain this"
    assert row.tool_names == "read_file"
