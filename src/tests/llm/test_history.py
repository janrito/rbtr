"""Tests for engine/history.py — history repair, demote_thinking,
and format-error detection.
"""

from __future__ import annotations

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from rbtr.llm.history import (
    demote_thinking,
    flatten_tool_exchanges,
    is_history_format_error,
    repair_dangling_tool_calls,
)

# ── demote_thinking ──────────────────────────────────────────────────


def test_demote_thinking_converts_to_text() -> None:
    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ThinkingPart(content="reasoning…", id="reasoning_content"),
                TextPart(content="hello"),
            ],
            model_name="test",
        ),
    ]

    cleaned = demote_thinking(history)
    assert len(cleaned) == 1
    response = cleaned[0]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 2
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "<thinking>\nreasoning…\n</thinking>"
    assert isinstance(response.parts[1], TextPart)
    assert response.parts[1].content == "hello"


def test_demote_thinking_drops_empty_thinking() -> None:
    history: list[ModelMessage] = [
        ModelResponse(
            parts=[ThinkingPart(content="", id="rs_123")],
            model_name="test",
        ),
    ]

    cleaned = demote_thinking(history)
    assert len(cleaned) == 0


def test_demote_thinking_preserves_non_responses() -> None:
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
    ]

    cleaned = demote_thinking(history)
    assert len(cleaned) == 1


# ── flatten_tool_exchanges ────────────────────────────────────────────


def test_flatten_tool_exchanges_converts_to_text() -> None:
    """Tool calls become text summaries; tool returns become user prompts."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[
                TextPart(content="Let me check…"),
                ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="file contents", tool_call_id="tc1"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="Here are the results.")]),
    ]
    cleaned = flatten_tool_exchanges(history)
    assert len(cleaned) == 4

    # ToolCallPart → TextPart with tool name.
    resp1 = cleaned[1]
    assert isinstance(resp1, ModelResponse)
    assert len(resp1.parts) == 2
    assert isinstance(resp1.parts[0], TextPart)
    assert resp1.parts[0].content == "Let me check…"
    assert isinstance(resp1.parts[1], TextPart)
    assert "read_file" in resp1.parts[1].content

    # ToolReturnPart → UserPromptPart with output preserved.
    req = cleaned[2]
    assert isinstance(req, ModelRequest)
    assert len(req.parts) == 1
    assert isinstance(req.parts[0], UserPromptPart)
    assert "file contents" in req.parts[0].content  # type: ignore[operator]


def test_flatten_tool_exchanges_preserves_all_messages() -> None:
    """No messages are dropped — tool-only messages become text messages."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="grep", args={"search": "TODO"}, tool_call_id="tc1"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="grep", content="line 42: TODO fix", tool_call_id="tc1"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="done")]),
    ]
    cleaned = flatten_tool_exchanges(history)
    assert len(cleaned) == 4

    # Tool-only response now has a TextPart.
    resp = cleaned[1]
    assert isinstance(resp, ModelResponse)
    assert isinstance(resp.parts[0], TextPart)
    assert "grep" in resp.parts[0].content

    # Tool return content preserved in user prompt.
    req = cleaned[2]
    assert isinstance(req, ModelRequest)
    assert isinstance(req.parts[0], UserPromptPart)
    assert "TODO fix" in req.parts[0].content  # type: ignore[operator]


def test_flatten_tool_exchanges_keeps_user_prompts_in_mixed_requests() -> None:
    """UserPromptParts in mixed requests are preserved alongside converted returns."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="tc1"),
                UserPromptPart(content="continue"),
            ]
        ),
    ]
    cleaned = flatten_tool_exchanges(history)
    assert len(cleaned) == 1
    req = cleaned[0]
    assert isinstance(req, ModelRequest)
    assert len(req.parts) == 2
    # Converted tool return.
    assert isinstance(req.parts[0], UserPromptPart)
    assert "read_file" in req.parts[0].content  # type: ignore[operator]
    # Original user prompt.
    assert isinstance(req.parts[1], UserPromptPart)
    assert req.parts[1].content == "continue"


# ── is_history_format_error ──────────────────────────────────────────


def test_is_history_format_error_invalid_id() -> None:
    exc = ModelHTTPError(
        400,
        "gpt-5.1-codex",
        body={
            "message": "Invalid 'input[6].id': 'reasoning_content'. "
            "Expected an ID that begins with 'rs'.",
        },
    )
    assert is_history_format_error(exc)


def test_is_history_format_error_missing_reasoning() -> None:
    exc = ModelHTTPError(
        400,
        "gpt-5-mini",
        body={
            "message": "Item 'fc_07f6' of type 'function_call' was "
            "provided without its required 'reasoning' item: 'rs_07f6'.",
        },
    )
    assert is_history_format_error(exc)


def test_is_history_format_error_rejects_unrelated() -> None:
    exc = ModelHTTPError(
        400,
        "gpt-4o",
        body={"message": "maximum context length exceeded"},
    )
    assert not is_history_format_error(exc)


def test_is_history_format_error_claude_tool_pairing() -> None:
    """Claude: tool_use IDs without matching tool_result blocks."""
    exc = ModelHTTPError(
        400,
        "claude-sonnet-4-6",
        body={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "messages.2: `tool_use` ids were found without "
                "`tool_result` blocks immediately after: "
                "call_XFAPJJDTOhbn3UngXP8OLS9b, call_f0SgvtE4qOFKr5UD61keJKyy. "
                "Each `tool_use` block must have a corresponding "
                "`tool_result` block in the next message.",
            },
        },
    )
    assert is_history_format_error(exc)


def test_is_history_format_error_orphan_tool_return() -> None:
    """Orphaned tool returns (from bad compaction) are format errors.

    These can occur when history from one provider is replayed to
    another.  The retry with simplified history strips all tool
    exchanges, which resolves the issue.
    """
    exc = ModelHTTPError(
        400,
        "gpt-5.3-codex",
        body={
            "message": "No tool call found for function call output "
            "with call_id call_2dSruMECzg5uxFmSBi4lC893.",
            "type": "invalid_request_error",
        },
    )
    assert is_history_format_error(exc)


def test_is_history_format_error_required_field() -> None:
    """Provider rejects messages with a missing required field.

    Some OpenAI-compatible endpoints require fields that PydanticAI
    omits (e.g. ``content`` on assistant messages containing only
    tool calls).  The pattern is kept general — ``required`` +
    ``field`` — to cover varying error formats across endpoints.
    """
    exc = ModelHTTPError(
        400,
        "some-model",
        body={
            "message": "The content field is a required field.",
        },
    )
    assert is_history_format_error(exc)


# ── repair_dangling_tool_calls ───────────────────────────────────────


def test_repair_empty_history() -> None:
    history, tools, _ = repair_dangling_tool_calls([])
    assert history == []
    assert tools == []


def test_repair_no_dangling_calls() -> None:
    """History ending with a ModelRequest is already well-formed."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi")]),
        ModelRequest(parts=[UserPromptPart(content="bye")]),
    ]
    repaired, tools, _ = repair_dangling_tool_calls(history)
    assert repaired is history
    assert tools == []


def test_repair_response_without_tool_calls() -> None:
    """A text-only ModelResponse doesn't need repair."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi")]),
    ]
    repaired, tools, _ = repair_dangling_tool_calls(history)
    assert repaired is history
    assert tools == []


def test_repair_dangling_at_end() -> None:
    """ModelResponse with tool calls at end of history → synthetic results appended."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[
                TextPart(content="Let me check..."),
                ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
                ToolCallPart(tool_name="grep", args={"search": "TODO"}, tool_call_id="tc2"),
            ]
        ),
    ]
    repaired, tools, _ = repair_dangling_tool_calls(history)
    assert len(repaired) == 3  # original 2 + synthetic request
    assert tools == ["read_file", "grep"]

    synthetic = repaired[-1]
    assert isinstance(synthetic, ModelRequest)
    tool_returns = [p for p in synthetic.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 2
    assert tool_returns[0].tool_name == "read_file"
    assert tool_returns[0].tool_call_id == "tc1"
    assert tool_returns[0].content == "(cancelled)"
    assert tool_returns[1].tool_name == "grep"
    assert tool_returns[1].tool_call_id == "tc2"


def test_repair_dangling_mid_history() -> None:
    """Dangling tool calls in the middle of history (e.g. after compaction)."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        # This response has tool calls but the next message is NOT tool results.
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="list_files", args={}, tool_call_id="tc1"),
            ]
        ),
        # A user prompt follows directly — tool results were lost.
        ModelRequest(parts=[UserPromptPart(content="try again")]),
        ModelResponse(parts=[TextPart(content="ok")]),
    ]
    repaired, tools, _ = repair_dangling_tool_calls(history)
    assert tools == ["list_files"]
    assert len(repaired) == 5  # 4 original + 1 synthetic

    # Synthetic request inserted after the dangling response.
    assert isinstance(repaired[2], ModelRequest)
    returns = [p for p in repaired[2].parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 1
    assert returns[0].tool_name == "list_files"
    assert returns[0].content == "(cancelled)"

    # Rest of history preserved.
    assert isinstance(repaired[3], ModelRequest)
    assert isinstance(repaired[4], ModelResponse)


def test_repair_tool_calls_with_results_untouched() -> None:
    """Tool calls followed by proper results are not touched."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="file contents", tool_call_id="tc1"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="done")]),
    ]
    repaired, tools, _ = repair_dangling_tool_calls(history)
    assert repaired is history
    assert tools == []


def test_repair_partial_results() -> None:
    """Some tool calls answered, others missing → only missing ones get synthetic results."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
                ToolCallPart(tool_name="grep", args={"search": "TODO"}, tool_call_id="tc2"),
                ToolCallPart(tool_name="diff", args={}, tool_call_id="tc3"),
            ]
        ),
        # Only tc1 completed before cancellation.
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="file contents", tool_call_id="tc1"),
            ]
        ),
    ]
    repaired, tools, new_msgs = repair_dangling_tool_calls(history)
    assert tools == ["grep", "diff"]
    # Original 3 messages + 1 synthetic for the 2 missing results.
    assert len(repaired) == 4

    # The existing partial results are preserved.
    assert isinstance(repaired[2], ModelRequest)
    existing = [p for p in repaired[2].parts if isinstance(p, ToolReturnPart)]
    assert len(existing) == 1
    assert existing[0].tool_call_id == "tc1"

    # Synthetic results appended for the missing calls.
    synthetic = repaired[3]
    assert isinstance(synthetic, ModelRequest)
    cancelled = [p for p in synthetic.parts if isinstance(p, ToolReturnPart)]
    assert len(cancelled) == 2
    assert cancelled[0].tool_call_id == "tc2"
    assert cancelled[0].content == "(cancelled)"
    assert cancelled[1].tool_call_id == "tc3"
    assert cancelled[1].content == "(cancelled)"

    # new_msgs contains only the synthetic messages (for persistence).
    assert len(new_msgs) == 1
    assert new_msgs[0] is synthetic


def test_repair_partial_results_mid_history() -> None:
    """Partial results in the middle of history — conversation continues after."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="read_file", args={}, tool_call_id="tc1"),
                ToolCallPart(tool_name="grep", args={}, tool_call_id="tc2"),
            ]
        ),
        # Only tc1 answered.
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="tc1"),
            ]
        ),
        # User continued with a new prompt.
        ModelRequest(parts=[UserPromptPart(content="try again")]),
        ModelResponse(parts=[TextPart(content="ok")]),
    ]
    repaired, tools, _ = repair_dangling_tool_calls(history)
    assert tools == ["grep"]
    assert len(repaired) == 6  # 5 original + 1 synthetic

    # Synthetic inserted after the partial results.
    synthetic = repaired[3]
    assert isinstance(synthetic, ModelRequest)
    cancelled = [p for p in synthetic.parts if isinstance(p, ToolReturnPart)]
    assert len(cancelled) == 1
    assert cancelled[0].tool_call_id == "tc2"

    # Rest of history preserved.
    assert isinstance(repaired[4], ModelRequest)  # "try again"
    assert isinstance(repaired[5], ModelResponse)  # "ok"


def test_repair_tool_returns_after_interleaved_user_prompt() -> None:
    """Tool returns appear later in history — user prompt interleaved between call and return.

    Reproduces a real scenario: the model issues tool calls, the user
    types a prompt before returns arrive, and the tool returns end up
    in a later message.  The repair must not inject synthetic cancelled
    results when the real returns exist further in the history.
    """
    history: list[ModelMessage] = [
        # User prompt.
        ModelRequest(parts=[UserPromptPart(content="review this PR")]),
        # Model responds with tool calls.
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="changed_files", args={}, tool_call_id="tc1"),
                ToolCallPart(tool_name="commit_log", args={}, tool_call_id="tc2"),
            ]
        ),
        # User types another prompt before tools complete.
        ModelRequest(parts=[UserPromptPart(content="also check the tests")]),
        # Model responds to the new prompt with its own tool call.
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="grep", args={"search": "test"}, tool_call_id="tc3"),
            ]
        ),
        # Tool returns for tc1, tc2 (from first response) AND tc3 arrive together.
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="changed_files", content="files", tool_call_id="tc1"),
                ToolReturnPart(tool_name="commit_log", content="log", tool_call_id="tc2"),
                ToolReturnPart(tool_name="grep", content="matches", tool_call_id="tc3"),
            ]
        ),
        # Model produces final text.
        ModelResponse(parts=[TextPart(content="Here are the findings.")]),
    ]
    repaired, tools, new_msgs = repair_dangling_tool_calls(history)

    # No repairs needed — all tool calls have matching returns.
    assert repaired is history
    assert tools == []
    assert new_msgs == []


def test_repair_returns_scattered_across_multiple_messages() -> None:
    """Tool returns split across several non-adjacent request messages.

    Reproduces the real scenario from session 019ca5be: multiple
    tool-calling rounds where returns for earlier calls appear in
    later request messages, interleaved with user prompts.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="start")]),
        # Round 1: 4 tool calls.
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="changed_files", args={}, tool_call_id="tc1"),
                ToolCallPart(tool_name="commit_log", args={}, tool_call_id="tc2"),
                ToolCallPart(tool_name="changed_symbols", args={}, tool_call_id="tc3"),
                ToolCallPart(tool_name="get_pr_discussion", args={}, tool_call_id="tc4"),
            ]
        ),
        # User prompt interleaved before returns.
        ModelRequest(parts=[UserPromptPart(content="review this")]),
        # Round 2: model makes its own call.
        ModelResponse(parts=[ToolCallPart(tool_name="diff", args={}, tool_call_id="tc5")]),
        # Returns for tc1-tc4 from round 1 appear here.
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="changed_files", content="files", tool_call_id="tc1"),
                ToolReturnPart(tool_name="commit_log", content="log", tool_call_id="tc2"),
                ToolReturnPart(tool_name="changed_symbols", content="syms", tool_call_id="tc3"),
                ToolReturnPart(tool_name="get_pr_discussion", content="disc", tool_call_id="tc4"),
            ]
        ),
        # Round 3: more calls.
        ModelResponse(parts=[ToolCallPart(tool_name="grep", args={}, tool_call_id="tc6")]),
        # Returns for tc5 and tc6 appear here.
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="diff", content="patch", tool_call_id="tc5"),
                ToolReturnPart(tool_name="grep", content="matches", tool_call_id="tc6"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="Done.")]),
    ]
    repaired, tools, new_msgs = repair_dangling_tool_calls(history)
    assert repaired is history
    assert tools == []
    assert new_msgs == []


def test_repair_idempotent_with_prior_synthetics() -> None:
    """Running repair on history that already has synthetic cancelled returns.

    The old bug: repair injected (cancelled) returns, saved them, then
    on next load re-injected more because the original calls still
    looked "dangling" (returns were further ahead).  The fix must be
    idempotent — existing synthetic returns count as answered.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="start")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="read_file", args={}, tool_call_id="tc1"),
                ToolCallPart(tool_name="grep", args={}, tool_call_id="tc2"),
            ]
        ),
        # User prompt interleaved.
        ModelRequest(parts=[UserPromptPart(content="continue")]),
        # Real returns exist later.
        ModelResponse(parts=[ToolCallPart(tool_name="diff", args={}, tool_call_id="tc3")]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="content", tool_call_id="tc1"),
                ToolReturnPart(tool_name="grep", content="matches", tool_call_id="tc2"),
                ToolReturnPart(tool_name="diff", content="patch", tool_call_id="tc3"),
            ]
        ),
        # Synthetic cancelled returns from a prior buggy repair run.
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="(cancelled)", tool_call_id="tc1"),
                ToolReturnPart(tool_name="grep", content="(cancelled)", tool_call_id="tc2"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="Done.")]),
    ]
    repaired, tools, new_msgs = repair_dangling_tool_calls(history)
    # No new repairs — both real returns and prior synthetics count.
    assert repaired is history
    assert tools == []
    assert new_msgs == []


def test_repair_mixed_matched_and_truly_missing() -> None:
    """Some calls have returns later in history, others are truly missing."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="start")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="read_file", args={}, tool_call_id="tc1"),
                ToolCallPart(tool_name="grep", args={}, tool_call_id="tc2"),
                ToolCallPart(tool_name="diff", args={}, tool_call_id="tc3"),
            ]
        ),
        # User prompt interleaved.
        ModelRequest(parts=[UserPromptPart(content="continue")]),
        # Only tc1 has a return later. tc2 and tc3 are truly missing.
        ModelResponse(parts=[ToolCallPart(tool_name="list_files", args={}, tool_call_id="tc4")]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="tc1"),
                ToolReturnPart(tool_name="list_files", content="files", tool_call_id="tc4"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="Done.")]),
    ]
    _repaired, tools, new_msgs = repair_dangling_tool_calls(history)
    # Only tc2 and tc3 should be repaired — tc1 has a real return later.
    assert sorted(tools) == ["diff", "grep"]
    assert len(new_msgs) == 1
