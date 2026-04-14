"""Tests for interrupted model response handling.

Covers detection of interrupted responses (`finish_reason` is
`length`, `content_filter`, or `error`) and the two recovery
paths:

- **Case 1:** Truncated tool-call args — repair in-place and
  replace the retry prompt with an informative message.
- **Case 2:** Interrupted text-only response — signal the caller
  to re-enter with a continuation prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    FinishReason,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.usage import RunUsage
from pytest_mock import MockerFixture

from rbtr.engine.core import Engine
from rbtr.llm.stream import (
    _check_interrupted_response,
    _replace_truncated_retry_prompts,
    _StreamResult,
    handle_llm,
)

# ── Shared data ──────────────────────────────────────────────────────

_TRUNCATED_JSON = '{"path": "notes.md", "new_text": "# Review\\nThis is a lo'
"""Truncated JSON string — EOF mid-value."""

_VALID_ARGS = {"path": "notes.md", "new_text": "done"}


def _response(
    *,
    finish_reason: FinishReason | None = "stop",
    parts: Sequence[ModelResponsePart] | None = None,
) -> ModelResponse:
    """Build a `ModelResponse` with the given finish reason and parts."""
    if parts is None:
        parts = [TextPart(content="Hello")]
    return ModelResponse(
        parts=parts,
        model_name="test-model",
        finish_reason=finish_reason,
    )


def _ok_result(**overrides: Any) -> _StreamResult:
    """Build a `_StreamResult` representing a normal completion."""
    defaults: dict[str, Any] = {
        "all_messages": [],
        "new_messages": [],
        "usage": RunUsage(requests=1, input_tokens=100, output_tokens=50),
        "limit_hit": False,
        "last_writer": None,
        "compact_needed": False,
        "interrupted": None,
    }
    defaults.update(overrides)
    return _StreamResult(**defaults)


def _interrupted_result() -> _StreamResult:
    """Build a `_StreamResult` with a text-only interruption."""
    return _ok_result(interrupted="Continue from where you stopped.")


# ── _check_interrupted_response: normal completions ─────────────────


@pytest.mark.parametrize(
    "finish_reason",
    ["stop", "tool_call", None],
    ids=["stop", "tool_call", "none"],
)
def test_normal_finish_returns_none(finish_reason: FinishReason | None) -> None:
    """`stop`, `tool_call`, and `None` are not interrupted."""
    resp = _response(finish_reason=finish_reason)
    assert _check_interrupted_response(resp) is None


# ── _check_interrupted_response: interrupted text-only ───────────────


@pytest.mark.parametrize(
    "finish_reason",
    ["length", "content_filter", "error"],
    ids=["length", "content_filter", "error"],
)
def test_interrupted_text_only(finish_reason: FinishReason) -> None:
    """Text-only response with interrupted finish reason is detected."""
    resp = _response(
        finish_reason=finish_reason,
        parts=[TextPart(content="partial output")],
    )
    result = _check_interrupted_response(resp)
    assert result is not None

    assert result.truncated_tool_ids == []
    assert len(result.message) > 0


# ── _check_interrupted_response: interrupted with valid tool args ────


def test_interrupted_with_valid_tool_args() -> None:
    """Interrupted response with parseable tool args has no truncated IDs."""
    resp = _response(
        finish_reason="length",
        parts=[
            ToolCallPart(tool_name="edit", args=_VALID_ARGS, tool_call_id="tc1"),
        ],
    )
    result = _check_interrupted_response(resp)
    assert result is not None

    assert result.truncated_tool_ids == []


# ── _check_interrupted_response: truncated tool args ─────────────────


@pytest.mark.parametrize(
    "finish_reason",
    ["length", "content_filter", "error"],
    ids=["length", "content_filter", "error"],
)
def test_truncated_tool_args_detected(finish_reason: FinishReason) -> None:
    """Truncated tool-call args are detected and IDs collected."""
    resp = _response(
        finish_reason=finish_reason,
        parts=[
            ToolCallPart(
                tool_name="edit",
                args=_TRUNCATED_JSON,
                tool_call_id="tc1",
            ),
        ],
    )
    result = _check_interrupted_response(resp)
    assert result is not None

    assert result.truncated_tool_ids == ["tc1"]


def test_truncated_tool_args_repaired_in_place() -> None:
    """After detection, truncated args are mutated to `{}` in place."""
    part = ToolCallPart(
        tool_name="edit",
        args=_TRUNCATED_JSON,
        tool_call_id="tc1",
    )
    resp = _response(finish_reason="length", parts=[part])

    _check_interrupted_response(resp)

    assert part.args_as_dict() == {}


def test_mixed_valid_and_truncated_tool_args() -> None:
    """Only the truncated tool call is reported; valid one is untouched."""
    valid_part = ToolCallPart(
        tool_name="read_file",
        args=_VALID_ARGS,
        tool_call_id="tc_ok",
    )
    broken_part = ToolCallPart(
        tool_name="edit",
        args=_TRUNCATED_JSON,
        tool_call_id="tc_bad",
    )
    resp = _response(
        finish_reason="length",
        parts=[valid_part, broken_part],
    )

    result = _check_interrupted_response(resp)

    assert result is not None

    assert result.truncated_tool_ids == ["tc_bad"]
    assert valid_part.args == _VALID_ARGS
    assert broken_part.args == {}


# ── _check_interrupted_response: messages per finish reason ──────────


def test_length_tool_message_mentions_token_limit() -> None:
    """The `length` tool-call message mentions output token limit."""
    resp = _response(
        finish_reason="length",
        parts=[
            ToolCallPart(tool_name="edit", args=_TRUNCATED_JSON, tool_call_id="tc1"),
        ],
    )
    result = _check_interrupted_response(resp)
    assert result is not None

    assert "token limit" in result.message.lower()


def test_content_filter_tool_message_mentions_filter() -> None:
    """The `content_filter` tool-call message mentions content filter."""
    resp = _response(
        finish_reason="content_filter",
        parts=[
            ToolCallPart(tool_name="edit", args=_TRUNCATED_JSON, tool_call_id="tc1"),
        ],
    )
    result = _check_interrupted_response(resp)
    assert result is not None

    assert "content filter" in result.message.lower()


def test_error_tool_message_mentions_error() -> None:
    """The `error` tool-call message mentions provider error."""
    resp = _response(
        finish_reason="error",
        parts=[
            ToolCallPart(tool_name="edit", args=_TRUNCATED_JSON, tool_call_id="tc1"),
        ],
    )
    result = _check_interrupted_response(resp)
    assert result is not None

    assert "error" in result.message.lower()


def test_length_text_message_mentions_continue() -> None:
    """The `length` text-continuation message tells the model to continue."""
    resp = _response(
        finish_reason="length",
        parts=[TextPart(content="partial")],
    )
    result = _check_interrupted_response(resp)
    assert result is not None

    assert "continue" in result.message.lower()


# ── _replace_truncated_retry_prompts ─────────────────────────────────


def test_replace_matching_retry_prompt() -> None:
    """Matching `RetryPromptPart` content is replaced with the message."""
    parts = [
        RetryPromptPart(
            content="1 validation error: missing field 'path'",
            tool_name="edit",
            tool_call_id="tc1",
        ),
    ]
    _replace_truncated_retry_prompts(
        parts,
        truncated_tool_ids={"tc1"},
        message="Your tool call was truncated.",
    )
    assert parts[0].content == "Your tool call was truncated."


def test_non_matching_retry_prompt_untouched() -> None:
    """RetryPromptPart for a different tool_call_id is not modified."""
    original_content = "1 validation error: missing field 'path'"
    parts = [
        RetryPromptPart(
            content=original_content,
            tool_name="other_tool",
            tool_call_id="tc_other",
        ),
    ]
    _replace_truncated_retry_prompts(
        parts,
        truncated_tool_ids={"tc1"},
        message="Your tool call was truncated.",
    )
    assert parts[0].content == original_content


def test_replace_only_matching_in_mixed_parts() -> None:
    """Only the matching RetryPromptPart is replaced; others are untouched."""
    matching = RetryPromptPart(
        content="missing fields",
        tool_name="edit",
        tool_call_id="tc_bad",
    )
    unrelated = RetryPromptPart(
        content="unknown tool name",
        tool_name="foo",
        tool_call_id="tc_ok",
    )
    parts = [matching, unrelated]

    _replace_truncated_retry_prompts(
        parts,
        truncated_tool_ids={"tc_bad"},
        message="Truncated!",
    )

    assert matching.content == "Truncated!"
    assert unrelated.content == "unknown tool name"


# ── Integration: text continuation via _stream_agent_inner ───────────


def test_text_continuation_retries_with_prompt(
    mocker: MockerFixture,
    config_path: Path,
    llm_engine: Engine,
) -> None:
    """An interrupted text-only response triggers a continuation call."""
    call_prompts: list[str | None] = []

    async def fake_do_stream(
        _ctx: object,
        _model: object,
        _deps: object,
        _settings: object,
        prompt: str | None,
        _history: object,
    ) -> _StreamResult:
        call_prompts.append(prompt)
        if len(call_prompts) == 1:
            return _interrupted_result()
        return _ok_result()

    mocker.patch("rbtr.llm.stream._do_stream", fake_do_stream)

    handle_llm(llm_engine._llm_context(), "analyse the code")

    assert len(call_prompts) == 2
    assert call_prompts[0] == "analyse the code"
    assert call_prompts[1] is not None
    assert "continue" in call_prompts[1].lower()


def test_text_continuation_stops_after_max_retries(
    mocker: MockerFixture,
    config_path: Path,
    llm_engine: Engine,
) -> None:
    """After `max_continuations` retries, the loop stops."""
    call_count = 0

    async def fake_do_stream(
        _ctx: object,
        _model: object,
        _deps: object,
        _settings: object,
        _prompt: object,
        _history: object,
    ) -> _StreamResult:
        nonlocal call_count
        call_count += 1
        return _interrupted_result()

    mocker.patch("rbtr.llm.stream._do_stream", fake_do_stream)

    handle_llm(llm_engine._llm_context(), "write the review")

    # 1 initial + max_continuations (default 2) = 3 total.
    assert call_count == 3


def test_text_continuation_not_triggered_on_normal_completion(
    mocker: MockerFixture,
    config_path: Path,
    llm_engine: Engine,
) -> None:
    """A normal completion does not trigger any continuation."""
    call_count = 0

    async def fake_do_stream(
        _ctx: object,
        _model: object,
        _deps: object,
        _settings: object,
        _prompt: object,
        _history: object,
    ) -> _StreamResult:
        nonlocal call_count
        call_count += 1
        return _ok_result()

    mocker.patch("rbtr.llm.stream._do_stream", fake_do_stream)

    handle_llm(llm_engine._llm_context(), "hello")

    assert call_count == 1
