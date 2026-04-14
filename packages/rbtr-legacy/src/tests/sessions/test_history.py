"""Tests for `rbtr.sessions.history` — history repair, demote_thinking,
and format-error detection.
"""

from __future__ import annotations

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pytest_cases import case, parametrize_with_cases

from rbtr_legacy.sessions.history import (
    consolidate_tool_returns,
    demote_thinking,
    flatten_tool_exchanges,
    is_history_format_error,
    repair_dangling_tool_calls,
    sanitize_tool_call_ids,
    strip_orphaned_tool_returns,
    validate_tool_call_args,
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

    result = demote_thinking(history)
    assert result.parts_demoted == 1
    assert len(result.history) == 1
    response = result.history[0]
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

    result = demote_thinking(history)
    assert result.parts_demoted == 0
    assert len(result.history) == 0


def test_demote_thinking_preserves_non_responses() -> None:
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
    ]

    result = demote_thinking(history)
    assert result.parts_demoted == 0
    assert len(result.history) == 1


# ── flatten_tool_exchanges ────────────────────────────────────────────


def test_flatten_tool_exchanges_converts_to_text() -> None:
    """Tool calls → text summaries, tool returns → user prompts, all messages preserved."""
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
    result = flatten_tool_exchanges(history)
    assert result.tool_calls_flattened == 1
    assert result.tool_returns_flattened == 1
    assert result.retry_prompts_dropped == 0
    assert len(result.history) == 4

    # ToolCallPart → TextPart preserving tool name and args.
    resp1 = result.history[1]
    assert isinstance(resp1, ModelResponse)
    assert isinstance(resp1.parts[1], TextPart)
    assert "read_file" in resp1.parts[1].content
    assert '"path"' in resp1.parts[1].content

    # ToolReturnPart → UserPromptPart preserving output.
    req = result.history[2]
    assert isinstance(req, ModelRequest)
    assert isinstance(req.parts[0], UserPromptPart)
    assert "file contents" in req.parts[0].content  # type: ignore[operator]


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
    result = flatten_tool_exchanges(history)
    assert result.tool_returns_flattened == 1
    assert len(result.history) == 1
    req = result.history[0]
    assert isinstance(req, ModelRequest)
    assert len(req.parts) == 2
    # Converted tool return.
    assert isinstance(req.parts[0], UserPromptPart)
    assert "read_file" in req.parts[0].content  # type: ignore[operator]
    # Original user prompt.
    assert isinstance(req.parts[1], UserPromptPart)
    assert req.parts[1].content == "continue"


def test_flatten_tool_exchanges_drops_retry_prompts() -> None:
    """RetryPromptParts are counted and dropped during flattening."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                RetryPromptPart(content="bad args", tool_call_id="tc1"),
                ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="tc2"),
            ]
        ),
    ]
    result = flatten_tool_exchanges(history)
    assert result.retry_prompts_dropped == 1
    assert result.tool_returns_flattened == 1
    # Only the converted tool return survives.
    req = result.history[0]
    assert isinstance(req, ModelRequest)
    assert len(req.parts) == 1
    assert isinstance(req.parts[0], UserPromptPart)


def test_flatten_empty_request_after_stripping() -> None:
    """A request with only RetryPromptParts is dropped entirely."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hi")]),
        ModelResponse(parts=[TextPart(content="hello")]),
        ModelRequest(parts=[RetryPromptPart(content="bad", tool_call_id="tc1")]),
    ]
    result = flatten_tool_exchanges(history)
    assert result.retry_prompts_dropped == 1
    # Empty request dropped — only the first two messages survive.
    assert len(result.history) == 2


# ── is_history_format_error ──────────────────────────────────────────


@case(tags=["format_error"])
def case_invalid_reasoning_id() -> str:
    """OpenAI rejects non-`rs` reasoning IDs."""
    return "Invalid 'input[6].id': 'reasoning_content'. Expected an ID that begins with 'rs'."


@case(tags=["format_error"])
def case_missing_reasoning_item() -> str:
    """OpenAI requires reasoning item paired with function_call."""
    return (
        "Item 'fc_07f6' of type 'function_call' was provided without its "
        "required 'reasoning' item: 'rs_07f6'."
    )


@case(tags=["format_error"])
def case_claude_tool_pairing() -> str:
    """Claude rejects unpaired `tool_use` IDs."""
    return (
        "messages.2: `tool_use` ids were found without `tool_result` blocks "
        "immediately after: call_XFAP, call_f0Sg."
    )


@case(tags=["format_error"])
def case_gemini_function_count() -> str:
    """Gemini requires equal function call and response counts."""
    return (
        "Please ensure that the number of function response parts is equal "
        "to the number of function call parts of the function call turn."
    )


@case(tags=["format_error"])
def case_orphaned_tool_return() -> str:
    """Provider rejects tool return with no matching call."""
    return "No tool call found for function call output with call_id call_2dSru."


@case(tags=["format_error"])
def case_extra_inputs() -> str:
    """Provider rejects extra/unexpected input fields."""
    return (
        "18 request validation errors: Extra inputs are not permitted, "
        "field: 'messages[1].rs_0cae3ab1ca0a8b8'"
    )


@case(tags=["format_error"])
def case_required_field() -> str:
    """Provider rejects missing required content field."""
    return "The content field is a required field."


@case(tags=["format_error"])
def case_tool_use_id_invalid_pattern() -> str:
    """Claude rejects tool_use IDs with invalid characters."""
    return "messages.0.content.2.tool_use.id: String should match pattern '^[a-zA-Z0-9_-]+'"


@case(tags=["not_format_error"])
def case_context_overflow() -> str:
    """Context overflow is not a format error."""
    return "maximum context length exceeded"


@case(tags=["not_format_error"])
def case_rate_limit() -> str:
    """Rate limiting is not a format error."""
    return "Rate limit exceeded. Please retry after 10 seconds."


@parametrize_with_cases("error_msg", cases=".", has_tag="format_error")
def test_is_history_format_error_positive(error_msg: str) -> None:
    """Error messages that indicate history format problems are detected."""
    exc = ModelHTTPError(400, "test-model", body={"message": error_msg})
    assert is_history_format_error(exc)


@parametrize_with_cases("error_msg", cases=".", has_tag="not_format_error")
def test_is_history_format_error_negative(error_msg: str) -> None:
    """Unrelated errors are not misidentified as format errors."""
    exc = ModelHTTPError(400, "test-model", body={"message": error_msg})
    assert not is_history_format_error(exc)


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
    """Some tool calls answered, others missing → synthetic results merged into existing request."""
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
    repaired, tools, _ = repair_dangling_tool_calls(history)
    assert tools == ["grep", "diff"]
    # 3 messages — synthetic returns merged into existing request.
    assert len(repaired) == 3

    # The merged request has existing + synthetic returns.
    merged = repaired[2]
    assert isinstance(merged, ModelRequest)
    returns = [p for p in merged.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 3
    assert returns[0].tool_call_id == "tc1"
    assert returns[0].content == "file contents"
    assert returns[1].tool_call_id == "tc2"
    assert returns[1].content == "(cancelled)"
    assert returns[2].tool_call_id == "tc3"
    assert returns[2].content == "(cancelled)"


def test_repair_partial_results_mid_history() -> None:
    """Partial results in the middle of history — synthetic merged, conversation continues."""
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
    # 5 messages — synthetic merged into existing request (no extra message).
    assert len(repaired) == 5

    # Merged request has original + synthetic returns.
    merged = repaired[2]
    assert isinstance(merged, ModelRequest)
    returns = [p for p in merged.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 2
    assert returns[0].tool_call_id == "tc1"
    assert returns[0].content == "ok"
    assert returns[1].tool_call_id == "tc2"
    assert returns[1].content == "(cancelled)"

    # Rest of history preserved.
    assert isinstance(repaired[3], ModelRequest)  # "try again"
    assert isinstance(repaired[4], ModelResponse)  # "ok"


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
    repaired, tools, _ = repair_dangling_tool_calls(history)

    # No repairs needed — all tool calls have matching returns.
    assert repaired is history
    assert tools == []


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
    repaired, tools, _ = repair_dangling_tool_calls(history)
    assert repaired is history
    assert tools == []


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
    repaired, tools, _ = repair_dangling_tool_calls(history)
    # No new repairs — both real returns and prior synthetics count.
    assert repaired is history
    assert tools == []


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
    _repaired, tools, _ = repair_dangling_tool_calls(history)
    # Only tc2 and tc3 should be repaired — tc1 has a real return later.
    assert sorted(tools) == ["diff", "grep"]


# ── consolidate_tool_returns ─────────────────────────────────────────


def test_consolidate_noop_on_clean_history() -> None:
    """History with properly paired tool exchanges is returned unchanged."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[ToolCallPart(tool_name="read_file", args={}, tool_call_id="tc1")],
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="tc1")],
        ),
        ModelResponse(parts=[TextPart(content="done")]),
    ]
    result = consolidate_tool_returns(history)
    assert result.history is history
    assert result.turns_fixed == 0


def test_consolidate_mixed_user_and_tool_parts() -> None:
    """Tool returns mixed with user prompt are separated into distinct requests."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[ToolCallPart(tool_name="read_file", args={}, tool_call_id="tc1")],
        ),
        # Mixed: tool return + user prompt in same request.
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="tc1"),
                UserPromptPart(content="also check tests"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="done")]),
    ]
    result = consolidate_tool_returns(history)
    assert result.turns_fixed == 1
    # Response followed by tool-return-only request, then user prompt request.
    assert len(result.history) == 5
    tool_req = result.history[2]
    assert isinstance(tool_req, ModelRequest)
    assert len(tool_req.parts) == 1
    assert isinstance(tool_req.parts[0], ToolReturnPart)

    user_req = result.history[3]
    assert isinstance(user_req, ModelRequest)
    assert isinstance(user_req.parts[0], UserPromptPart)


def test_consolidate_scattered_returns() -> None:
    """Tool returns spread across multiple requests are consolidated."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="read_file", args={}, tool_call_id="tc1"),
                ToolCallPart(tool_name="grep", args={}, tool_call_id="tc2"),
            ],
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="tc1")],
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="grep", content="found", tool_call_id="tc2")],
        ),
        ModelResponse(parts=[TextPart(content="done")]),
    ]
    result = consolidate_tool_returns(history)
    assert result.turns_fixed == 1
    # Scattered returns consolidated into one request.
    assert len(result.history) == 4
    tool_req = result.history[2]
    assert isinstance(tool_req, ModelRequest)
    returns = [p for p in tool_req.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 2
    assert returns[0].tool_call_id == "tc1"
    assert returns[1].tool_call_id == "tc2"


def test_consolidate_empty_history() -> None:
    """Empty history is returned unchanged."""
    result = consolidate_tool_returns([])
    assert result.history == []
    assert result.turns_fixed == 0


def test_consolidate_missing_return_injects_synthetic() -> None:
    """Truly missing returns get a synthetic `(cancelled)` placeholder."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="read_file", args={}, tool_call_id="tc1"),
                ToolCallPart(tool_name="grep", args={}, tool_call_id="tc2"),
            ],
        ),
        # Only tc1 has a return — tc2 is missing entirely.
        ModelRequest(
            parts=[ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="tc1")],
        ),
        ModelResponse(parts=[TextPart(content="done")]),
    ]
    result = consolidate_tool_returns(history)
    assert result.turns_fixed == 1
    tool_req = result.history[2]
    assert isinstance(tool_req, ModelRequest)
    returns = [p for p in tool_req.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 2
    assert returns[0].tool_call_id == "tc1"
    assert returns[1].tool_call_id == "tc2"
    assert returns[1].content == "(cancelled)"


# ── Level-0 preventive repairs ───────────────────────────────────────
#
# Shared dataset: a conversation started on Gemini (IDs with dots
# and colons), then switched to Claude (IDs with toolu_ prefix).
# One Gemini tool call has corrupt args from a streaming failure.

# Gemini turn — two tool calls with Gemini-style IDs.
_GEMINI_RESPONSE = ModelResponse(
    parts=[
        ToolCallPart(
            tool_name="read_file",
            args={"path": "src/main.py"},
            tool_call_id="functions.read_file:3",
        ),
        ToolCallPart(
            tool_name="grep",
            args='{"pattern": "TODO",\n<parameter name="path": "src/"}',
            tool_call_id="functions.grep:4",
        ),
    ],
    model_name="google/gemini-2.5-pro",
)
_GEMINI_RETURNS = ModelRequest(
    parts=[
        ToolReturnPart(
            tool_name="read_file",
            content="def main(): ...",
            tool_call_id="functions.read_file:3",
        ),
        ToolReturnPart(
            tool_name="grep",
            content="src/main.py:10: # TODO: fix",
            tool_call_id="functions.grep:4",
        ),
    ]
)

# Claude turn — valid IDs.
_CLAUDE_RESPONSE = ModelResponse(
    parts=[
        ToolCallPart(
            tool_name="read_file",
            args={"path": "src/utils.py"},
            tool_call_id="toolu_01ABC",
        ),
    ],
    model_name="claude/claude-sonnet-4-6",
)
_CLAUDE_RETURN = ModelRequest(
    parts=[
        ToolReturnPart(
            tool_name="read_file",
            content="def helper(): ...",
            tool_call_id="toolu_01ABC",
        ),
    ]
)

_MIXED_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content="review src/main.py")]),
    _GEMINI_RESPONSE,
    _GEMINI_RETURNS,
    ModelResponse(parts=[TextPart(content="Found a TODO.")], model_name="google/gemini-2.5-pro"),
    ModelRequest(parts=[UserPromptPart(content="check utils too")]),
    _CLAUDE_RESPONSE,
    _CLAUDE_RETURN,
    ModelResponse(parts=[TextPart(content="Looks clean.")], model_name="claude/claude-sonnet-4-6"),
]


# ── sanitize_tool_call_ids ───────────────────────────────────────────


def test_sanitize_skips_claude_ids() -> None:
    """Claude-style IDs (`toolu_*`) are already valid."""
    history: list[ModelMessage] = _MIXED_HISTORY[4:]  # Claude turns only.
    result, bad_ids = sanitize_tool_call_ids(history)
    assert bad_ids == []
    assert result is history


def test_sanitize_fixes_gemini_ids_preserves_pairing() -> None:
    """Gemini `functions.name:N` IDs are sanitized; pairing holds."""
    result, bad_ids = sanitize_tool_call_ids(list(_MIXED_HISTORY))
    assert bad_ids == ["functions.grep:4", "functions.read_file:3"]

    # Response side.
    resp = result[1]
    assert isinstance(resp, ModelResponse)
    assert resp.parts[0].tool_call_id == "functions_read_file_3"  # type: ignore[union-attr]
    assert resp.parts[1].tool_call_id == "functions_grep_4"  # type: ignore[union-attr]

    # Return side — same IDs.
    req = result[2]
    assert isinstance(req, ModelRequest)
    assert req.parts[0].tool_call_id == "functions_read_file_3"  # type: ignore[union-attr]
    assert req.parts[1].tool_call_id == "functions_grep_4"  # type: ignore[union-attr]

    # Claude IDs untouched.
    claude_resp = result[5]
    assert isinstance(claude_resp, ModelResponse)
    assert claude_resp.parts[0].tool_call_id == "toolu_01ABC"  # type: ignore[union-attr]


def test_sanitize_handles_retry_prompt_part() -> None:
    """`RetryPromptPart` IDs are sanitized alongside tool returns."""
    history: list[ModelMessage] = [
        ModelResponse(
            parts=[ToolCallPart(tool_name="grep", args={}, tool_call_id="functions.grep:7")],
        ),
        ModelRequest(
            parts=[RetryPromptPart(content="try again", tool_call_id="functions.grep:7")],
        ),
    ]
    result, bad_ids = sanitize_tool_call_ids(history)
    assert bad_ids == ["functions.grep:7"]

    req = result[1]
    assert isinstance(req, ModelRequest)
    retry = req.parts[0]
    assert isinstance(retry, RetryPromptPart)
    assert retry.tool_call_id == "functions_grep_7"


# ── validate_tool_call_args ──────────────────────────────────────────


def test_validate_args_skips_valid() -> None:
    """Well-formed args (dict or valid JSON string) are untouched."""
    history: list[ModelMessage] = _MIXED_HISTORY[4:]  # Claude turns only.
    repaired = validate_tool_call_args(history)
    assert repaired == []


def test_validate_args_repairs_corrupt_gemini_streaming() -> None:
    """Corrupt args from a Gemini streaming failure are replaced with `{}`."""
    history = list(_MIXED_HISTORY)
    repaired = validate_tool_call_args(history)
    assert repaired == [("grep", "functions.grep:4")]

    resp = history[1]
    assert isinstance(resp, ModelResponse)
    # Corrupt part repaired.
    grep_part = resp.parts[1]
    assert isinstance(grep_part, ToolCallPart)
    assert grep_part.args == {}
    # Valid part untouched.
    read_part = resp.parts[0]
    assert isinstance(read_part, ToolCallPart)
    assert read_part.args == {"path": "src/main.py"}


# ── strip_orphaned_tool_returns ──────────────────────────────────────


def test_strip_orphaned_returns_removes_unmatched() -> None:
    """Returns with no matching call are stripped from the request."""

    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="f", content="orphan", tool_call_id="gone"),
                UserPromptPart(content="continue"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="ok")], model_name="test"),
    ]
    repaired, stripped = strip_orphaned_tool_returns(history)

    assert stripped == ["gone"]
    assert len(repaired) == 2
    # The orphaned return was stripped; UserPromptPart survives.
    req = repaired[0]
    assert isinstance(req, ModelRequest)
    assert len(req.parts) == 1
    assert isinstance(req.parts[0], UserPromptPart)


def test_strip_orphaned_returns_preserves_matched() -> None:
    """Returns with a matching call are kept intact."""

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="read a.py")]),
        ModelResponse(
            parts=[
                TextPart(content="Reading."),
                ToolCallPart(tool_name="read_file", args={}, tool_call_id="c1"),
            ],
            model_name="test",
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="c1")]
        ),
        ModelResponse(parts=[TextPart(content="done")], model_name="test"),
    ]
    repaired, stripped = strip_orphaned_tool_returns(history)

    assert stripped == []
    assert repaired == history


def test_strip_orphaned_returns_noop_on_clean_history() -> None:
    """Clean history passes through unchanged."""

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi")], model_name="test"),
    ]
    repaired, stripped = strip_orphaned_tool_returns(history)

    assert stripped == []
    assert repaired == history


# ── Smoke: all history shapes are clean ──────────────────────────────


@parametrize_with_cases("history", cases="tests.sessions.case_histories")
def test_clean_history_needs_no_repair(history: list[ModelMessage]) -> None:
    """Every history case passes through all repairs unchanged."""
    history = list(history)

    stripped, stripped_ids = strip_orphaned_tool_returns(history)
    assert stripped_ids == [], "orphaned returns found"

    repaired, tools, _ = repair_dangling_tool_calls(stripped)
    assert tools == [], "dangling calls found"

    assert repaired == history, "repair changed the history"
