"""Tests for engine/llm.py — event emission, usage tracking, error
classification, and overflow/retry behaviour.

Tests target observable outcomes (events emitted, state mutated,
handlers retried) rather than internal async iteration mechanics.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
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
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage

from rbtr.config import config
from rbtr.engine.core import Engine
from rbtr.events import CompactionStarted, ToolCallFinished, ToolCallStarted
from rbtr.llm.context import LLMContext
from rbtr.llm.costs import record_run_usage
from rbtr.llm.stream import (
    _auto_compact_on_overflow,
    _emit_tool_event,
    _update_live_usage,
    handle_llm,
)
from rbtr.sessions.incidents import (
    FailedAttempt,
    FailureKind,
    HistoryRepair,
    IncidentOutcome,
    RecoveryStrategy,
)
from rbtr.sessions.kinds import FragmentKind
from tests.engine.builders import _turns
from tests.helpers import TestProvider, drain

# ── Shared data ──────────────────────────────────────────────────────

# A minimal FunctionToolCallEvent-like object. We construct real
# pydantic-ai event objects via their public constructors.


def _make_tool_call_event(
    tool_name: str, args: str | dict[str, Any] | None
) -> FunctionToolCallEvent:
    """Build a FunctionToolCallEvent with the given args."""

    part = ToolCallPart(
        tool_name=tool_name,
        args=args,
        tool_call_id="call_1",
    )
    return FunctionToolCallEvent(part=part)


def _make_tool_result_event(tool_name: str, content: str) -> FunctionToolResultEvent:
    """Build a FunctionToolResultEvent with a ToolReturnPart."""

    result = ToolReturnPart(
        tool_name=tool_name,
        content=content,
        tool_call_id="call_1",
    )
    return FunctionToolResultEvent(result=result)


def _make_tool_retry_event(tool_name: str, content: str) -> FunctionToolResultEvent:
    """Build a FunctionToolResultEvent with a RetryPromptPart (retry)."""

    result = RetryPromptPart(
        tool_name=tool_name,
        content=content,
        tool_call_id="call_1",
    )
    return FunctionToolResultEvent(result=result)


# ── _emit_tool_event ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("tool_name", "args", "expected_args"),
    [
        ("read_file", {"path": "foo.py"}, '"path"'),
        ("search", "hello world", "hello world"),
        ("list_files", None, ""),
    ],
    ids=["dict_args", "str_args", "none_args"],
)
def test_emit_tool_call(
    engine: Engine,
    llm_ctx: LLMContext,
    tool_name: str,
    args: str | dict[str, str] | None,
    expected_args: str,
) -> None:
    """Tool call events are emitted with correctly formatted args."""
    _emit_tool_event(llm_ctx, _make_tool_call_event(tool_name, args))
    events = drain(engine.events)
    assert len(events) == 1
    assert isinstance(events[0], ToolCallStarted)
    assert events[0].tool_name == tool_name
    assert expected_args in events[0].args


def test_emit_tool_result_normal(engine: Engine, llm_ctx: LLMContext) -> None:
    """Normal tool result emits content text."""
    _emit_tool_event(llm_ctx, _make_tool_result_event("read_file", "file contents here"))
    events = drain(engine.events)
    assert len(events) == 1
    assert isinstance(events[0], ToolCallFinished)
    assert events[0].result == "file contents here"


def test_emit_tool_result_truncation(engine: Engine, llm_ctx: LLMContext) -> None:
    """Long results are truncated at tool_max_chars."""

    long_content = "x" * (config.tui.tool_max_chars + 100)
    _emit_tool_event(llm_ctx, _make_tool_result_event("read_file", long_content))
    events = drain(engine.events)
    assert isinstance(events[0], ToolCallFinished)
    assert len(events[0].result) == config.tui.tool_max_chars + 1  # +1 for "…"
    assert events[0].result.endswith("…")


def test_emit_tool_result_retry(engine: Engine, llm_ctx: LLMContext) -> None:
    """Retry prompt result emits error with the failure message."""
    _emit_tool_event(llm_ctx, _make_tool_retry_event("read_file", "tool failed, try again"))
    events = drain(engine.events)
    assert isinstance(events[0], ToolCallFinished)
    assert events[0].result == ""
    assert events[0].error == "tool failed, try again"


def test_emit_tool_result_success_has_no_error(engine: Engine, llm_ctx: LLMContext) -> None:
    """Successful tool result has no error field set."""
    _emit_tool_event(llm_ctx, _make_tool_result_event("read_file", "file contents"))
    events = drain(engine.events)
    assert isinstance(events[0], ToolCallFinished)
    assert events[0].error is None


# ── record_run_usage ────────────────────────────────────────────────────


def _make_run_usage(*, input_tokens: int = 100, output_tokens: int = 50) -> RunUsage:
    return RunUsage(
        requests=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


@pytest.mark.parametrize(
    (
        "messages",
        "input_tokens",
        "output_tokens",
        "expected_cost",
        "expected_input",
        "expected_output",
    ),
    [
        ([], 100, 50, 0.0, 100, 50),
        (
            [
                ModelResponse(
                    parts=[TextPart(content="hello")], model_name="claude-sonnet-4-20250514"
                )
            ],
            500,
            200,
            None,
            500,
            200,
        ),
        (
            [ModelResponse(parts=[TextPart(content="hello")], model_name=None)],
            100,
            50,
            0.0,
            100,
            50,
        ),
    ],
    ids=["no_messages", "with_response", "no_model_name"],
)
def testrecord_run_usage_basics(
    llm_ctx: LLMContext,
    messages: list[ModelMessage],
    input_tokens: int,
    output_tokens: int,
    expected_cost: float | None,
    expected_input: int,
    expected_output: int,
) -> None:
    """Basic record_run_usage scenarios: empty, with response, no model name."""
    record_run_usage(
        llm_ctx, messages, _make_run_usage(input_tokens=input_tokens, output_tokens=output_tokens)
    )
    if expected_cost is not None:
        assert llm_ctx.state.usage.total_cost == expected_cost
    assert llm_ctx.state.usage.input_tokens == expected_input
    assert llm_ctx.state.usage.output_tokens == expected_output


# ── record_run_usage: context tracking ─────────────────────────────────

# Provider presets for building realistic ModelResponse objects.
_PROVIDERS: dict[str, tuple[str, str, str]] = {
    # rbtr model prefix → (pydantic-ai model_name, provider_name, provider_url)
    "claude": ("claude-sonnet-4-20250514", "anthropic", "https://api.anthropic.com"),
    "openai": ("gpt-4o", "openai", "https://api.openai.com/v1"),
}


def _make_provider_response(
    provider: str = "claude",
    *,
    input_tokens: int = 5000,
    output_tokens: int = 200,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> ModelResponse:
    """Build a ModelResponse mimicking what a real streaming run produces."""

    model_name, provider_name, provider_url = _PROVIDERS[provider]
    return ModelResponse(
        parts=[TextPart(content="response")],
        model_name=model_name,
        provider_name=provider_name,
        provider_url=provider_url,
        usage=RequestUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
    )


@pytest.mark.parametrize(
    ("model_name", "provider", "expected_window"),
    [
        ("claude/claude-sonnet-4-20250514", "claude", 200_000),
        ("openai/gpt-4o", "openai", 128_000),
    ],
)
def testrecord_run_usage_sets_context_window(
    model_name: str,
    provider: str,
    expected_window: int,
    engine: Engine,
    llm_ctx: LLMContext,
) -> None:
    """record_run_usage resolves the real context window for built-in providers."""
    engine.state.model_name = model_name
    response = _make_provider_response(provider, input_tokens=5000, output_tokens=200)
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        response,
    ]

    record_run_usage(llm_ctx, messages, _make_run_usage(input_tokens=5000, output_tokens=200))

    assert engine.state.usage.context_window == expected_window
    assert engine.state.usage.context_window_known is True
    assert engine.state.usage.last_input_tokens == 5000


def testrecord_run_usage_last_input_tokens_from_response(
    engine: Engine, llm_ctx: LLMContext
) -> None:
    """last_input_tokens comes from the last ModelResponse.usage, not cumulative RunUsage."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    # Simulate a multi-request run (tool calls):
    # RunUsage is cumulative across requests, but we want the last request's value.
    response1 = _make_provider_response(input_tokens=5000)
    response2 = _make_provider_response(input_tokens=8000)
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        response1,
        ModelRequest(parts=[UserPromptPart(content="tool result")]),
        response2,
    ]
    # RunUsage cumulative = 5000 + 8000 = 13000
    record_run_usage(llm_ctx, messages, _make_run_usage(input_tokens=13000, output_tokens=400))

    # last_input_tokens should be the last response's value, not cumulative
    assert engine.state.usage.last_input_tokens == 8000
    assert engine.state.usage.input_tokens == 13000


def testrecord_run_usage_context_used_pct(engine: Engine, llm_ctx: LLMContext) -> None:
    """context_used_pct uses the real context window, not the 128 k default."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    response = _make_provider_response(input_tokens=100_000)
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        response,
    ]

    record_run_usage(llm_ctx, messages, _make_run_usage(input_tokens=100_000))

    # 100k / 200k = 50%, NOT 100k / 128k = 78%
    assert engine.state.usage.context_window == 200_000
    assert engine.state.usage.context_used_pct == pytest.approx(50.0)


def testrecord_run_usage_cache_tokens(engine: Engine, llm_ctx: LLMContext) -> None:
    """Cache tokens are tracked alongside the context window."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    response = _make_provider_response(
        input_tokens=5000,
        cache_read_tokens=3000,
        cache_write_tokens=1000,
    )
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        response,
    ]
    usage = RunUsage(
        requests=1,
        input_tokens=5000,
        output_tokens=200,
        cache_read_tokens=3000,
        cache_write_tokens=1000,
    )

    record_run_usage(llm_ctx, messages, usage)

    assert engine.state.usage.cache_read_tokens == 3000
    assert engine.state.usage.cache_write_tokens == 1000
    assert engine.state.usage.context_window == 200_000


def testrecord_run_usage_multi_turn_context_pct(engine: Engine, llm_ctx: LLMContext) -> None:
    """Context % stays correct across multiple turns."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"

    # Turn 1
    resp1 = _make_provider_response(input_tokens=10_000)
    msgs1: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="turn 1")]),
        resp1,
    ]
    record_run_usage(llm_ctx, msgs1, _make_run_usage(input_tokens=10_000))

    assert engine.state.usage.context_window == 200_000
    assert engine.state.usage.context_used_pct == pytest.approx(5.0)

    # Turn 2 — context grows
    resp2 = _make_provider_response(input_tokens=50_000)
    msgs2: list[ModelMessage] = [
        *msgs1,
        ModelRequest(parts=[UserPromptPart(content="turn 2")]),
        resp2,
    ]
    new_msgs2 = msgs2[len(msgs1) :]
    record_run_usage(llm_ctx, new_msgs2, _make_run_usage(input_tokens=50_000))

    assert engine.state.usage.context_window == 200_000  # stays correct
    assert engine.state.usage.context_used_pct == pytest.approx(25.0)


# ── _update_live_usage: mid-run context tracking ─────────────────────


def test_update_live_usage_sets_context_window(engine: Engine, llm_ctx: LLMContext) -> None:
    """_update_live_usage resolves the context window during streaming."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine.state.usage.snapshot_base()

    response = _make_provider_response(input_tokens=10_000, output_tokens=500)
    run_usage = RunUsage(requests=1, input_tokens=10_000, output_tokens=500)

    _update_live_usage(llm_ctx, run_usage, response)

    assert engine.state.usage.last_input_tokens == 10_000
    assert engine.state.usage.context_window == 200_000
    assert engine.state.usage.context_window_known is True
    # 10k / 200k = 5%, NOT 10k / 128k = 7.8%
    assert engine.state.usage.context_used_pct == pytest.approx(5.0)


def test_update_live_usage_accumulates_over_base(engine: Engine, llm_ctx: LLMContext) -> None:
    """Live usage correctly adds to the baseline from previous runs."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"

    # Simulate a completed first run.
    engine.state.usage.record_run(
        input_tokens=5000,
        output_tokens=200,
        context_window=200_000,
    )
    engine.state.usage.snapshot_base()  # start of second run

    response = _make_provider_response(input_tokens=8000, output_tokens=300)
    run_usage = RunUsage(requests=1, input_tokens=8000, output_tokens=300)

    _update_live_usage(llm_ctx, run_usage, response)

    # Lifetime totals: 5000 + 8000 = 13000
    assert engine.state.usage.input_tokens == 13_000
    assert engine.state.usage.output_tokens == 500
    # But context % uses only last request's tokens
    assert engine.state.usage.last_input_tokens == 8000
    assert engine.state.usage.context_used_pct == pytest.approx(4.0)


def _make_http_error(status: HTTPStatus, body: str) -> ModelHTTPError:
    """Construct a ModelHTTPError with given status and body text."""
    return ModelHTTPError(status, "test-model", body)


def _failure_incidents(engine: Engine) -> list[FailedAttempt]:
    """Return deserialised `FailedAttempt` rows for the session."""
    rows = engine.store._con.execute(
        "SELECT data_json FROM fragments WHERE fragment_kind = ? AND session_id = ?",
        [FragmentKind.LLM_ATTEMPT_FAILED.value, engine.state.session_id],
    ).fetchall()
    return [FailedAttempt.model_validate_json(r["data_json"]) for r in rows]


def _repair_incidents(engine: Engine) -> list[HistoryRepair]:
    """Return deserialised `HistoryRepair` rows for the session."""
    rows = engine.store._con.execute(
        "SELECT data_json FROM fragments WHERE fragment_kind = ? AND session_id = ?",
        [FragmentKind.LLM_HISTORY_REPAIR.value, engine.state.session_id],
    ).fetchall()
    return [HistoryRepair.model_validate_json(r["data_json"]) for r in rows]


def _failed_request_rows(engine: Engine) -> list[Any]:
    """Return fragment rows for failed REQUEST_MESSAGE rows."""
    return engine.store._con.execute(
        "SELECT * FROM fragments WHERE fragment_kind = ? AND status = 'failed' AND session_id = ?",
        [FragmentKind.REQUEST_MESSAGE.value, engine.state.session_id],
    ).fetchall()


# ── handle_llm: effort-unsupported retry ─────────────────────────────


def test_handle_llm_retries_without_effort_on_rejection(
    llm_engine: Engine,
    test_provider: TestProvider,
) -> None:
    """handle_llm retries without effort when the model rejects it."""
    call_count = 0

    async def _reject_effort(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_http_error(
                HTTPStatus.BAD_REQUEST, "This model does not support the effort parameter."
            )
        yield "recovered"

    test_provider.set_model(FunctionModel(stream_function=_reject_effort))

    handle_llm(llm_engine._llm_context(), "test question")

    assert call_count == 2
    assert llm_engine.state.effort_supported is False

    # Incident: failed request + attempt-failed row with correct metadata.
    failed = _failed_request_rows(llm_engine)
    assert len(failed) == 1

    incidents = _failure_incidents(llm_engine)
    assert len(incidents) == 1
    assert incidents[0].failure_kind == FailureKind.EFFORT_UNSUPPORTED
    assert incidents[0].strategy == RecoveryStrategy.EFFORT_OFF
    assert incidents[0].outcome == IncidentOutcome.RECOVERED

    # User prompt survived recovery.
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert any(
        isinstance(p, UserPromptPart) and p.content == "test question"
        for m in loaded
        if isinstance(m, ModelRequest)
        for p in m.parts
    )
    assert incidents[0].turn_id == failed[0]["message_id"]


# ── _auto_compact_on_overflow ────────────────────────────────────────


def test_auto_compact_on_overflow_compacts_and_retries(
    llm_engine: Engine,
    test_provider: TestProvider,
) -> None:
    """Overflow handler compacts history then retries via handle_llm."""

    _seed = _turns(5)
    llm_engine._sync_store_context()
    llm_engine.store.save_messages(llm_engine.state.session_id, _seed)
    llm_engine.state.usage.context_window = 200_000

    _auto_compact_on_overflow(llm_engine._llm_context(), "my question")

    events = drain(llm_engine.events)
    assert any(isinstance(e, CompactionStarted) for e in events)


# ── handle_llm context overflow integration ──────────────────────────


def test_handle_llm_context_overflow_triggers_compact(
    llm_engine: Engine,
    test_provider: TestProvider,
) -> None:
    """handle_llm auto-compacts on context overflow and retries."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="analyse the codebase")]),
        ModelResponse(parts=[TextPart(content="I found 3 files so far")], model_name="test"),
        ModelRequest(parts=[UserPromptPart(content="turn 2")]),
        ModelResponse(parts=[TextPart(content="resp 2")], model_name="test"),
    ]
    llm_engine.store.save_messages(llm_engine.state.session_id, messages)
    llm_engine.state.usage.context_window = 200_000

    # First call overflows, pipeline compacts and retries
    call_count = 0

    async def _overflow_then_ok(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_http_error(HTTPStatus.BAD_REQUEST, "maximum context length exceeded")
        yield "recovered"

    test_provider.set_model(FunctionModel(stream_function=_overflow_then_ok))

    handle_llm(llm_engine._llm_context(), "test question")

    incidents = _failure_incidents(llm_engine)
    assert len(incidents) == 1
    assert incidents[0].failure_kind == FailureKind.OVERFLOW
    assert incidents[0].strategy == RecoveryStrategy.COMPACT_THEN_RETRY
    assert incidents[0].outcome == IncidentOutcome.RECOVERED

    # User prompt survived recovery.
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert any(
        isinstance(p, UserPromptPart) and p.content == "test question"
        for m in loaded
        if isinstance(m, ModelRequest)
        for p in m.parts
    )


# ── ValueError from corrupt tool-call args ───────────────────────────


def test_handle_llm_retries_on_corrupt_tool_args(
    llm_engine: Engine,
    test_provider: TestProvider,
) -> None:
    """handle_llm retries with simplified history on corrupt tool-call args."""
    call_count = 0

    async def _corrupt_then_ok(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("key must be a string at line 2 column 1")
        yield "recovered"

    test_provider.set_model(FunctionModel(stream_function=_corrupt_then_ok))

    handle_llm(llm_engine._llm_context(), "show my notes")

    incidents = _failure_incidents(llm_engine)
    assert len(incidents) == 1
    assert incidents[0].failure_kind == FailureKind.TOOL_ARGS
    assert incidents[0].strategy == RecoveryStrategy.SIMPLIFY_HISTORY
    assert incidents[0].outcome == IncidentOutcome.RECOVERED

    # User prompt survived recovery.
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert any(
        isinstance(p, UserPromptPart) and p.content == "show my notes"
        for m in loaded
        if isinstance(m, ModelRequest)
        for p in m.parts
    )


def test_handle_llm_retries_on_type_error(
    llm_engine: Engine,
    test_provider: TestProvider,
) -> None:
    """handle_llm retries with simplified history on TypeError."""
    call_count = 0

    async def _type_error_then_ok(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TypeError("'NoneType' object is not subscriptable")
        yield "recovered"

    test_provider.set_model(FunctionModel(stream_function=_type_error_then_ok))

    handle_llm(llm_engine._llm_context(), "hello")

    incidents = _failure_incidents(llm_engine)
    assert len(incidents) == 1
    assert incidents[0].failure_kind == FailureKind.TYPE_ERROR
    assert incidents[0].strategy == RecoveryStrategy.SIMPLIFY_HISTORY
    assert incidents[0].outcome == IncidentOutcome.RECOVERED

    # User prompt survived recovery.
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert any(
        isinstance(p, UserPromptPart) and p.content == "hello"
        for m in loaded
        if isinstance(m, ModelRequest)
        for p in m.parts
    )


# ── handle_llm: history-format retry ────────────────────────────────


def test_handle_llm_retries_on_history_format_error(
    llm_engine: Engine,
    test_provider: TestProvider,
) -> None:
    """handle_llm retries with simplified history on provider format rejection.

    When the provider rejects history due to reasoning IDs, unpaired
    tool calls, or other provider-specific metadata, `handle_llm`
    retries with `history_repair_level=1`.
    """

    call_count = 0

    async def _format_error_then_ok(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_http_error(
                HTTPStatus.BAD_REQUEST,
                "tool_use ids ['tool_1'] without matching tool_result blocks",
            )
        yield "recovered"

    test_provider.set_model(FunctionModel(stream_function=_format_error_then_ok))

    handle_llm(llm_engine._llm_context(), "continue analysis")

    # Incident: history_format failure → first retry uses consolidation.
    failed = _failed_request_rows(llm_engine)
    assert len(failed) == 1

    incidents = _failure_incidents(llm_engine)
    assert len(incidents) == 1
    assert incidents[0].failure_kind == FailureKind.HISTORY_FORMAT
    assert incidents[0].strategy == RecoveryStrategy.CONSOLIDATE_TOOL_RETURNS
    assert incidents[0].outcome == IncidentOutcome.RECOVERED

    # User prompt survived recovery.
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert any(
        isinstance(p, UserPromptPart) and p.content == "continue analysis"
        for m in loaded
        if isinstance(m, ModelRequest)
        for p in m.parts
    )
    assert incidents[0].status_code == HTTPStatus.BAD_REQUEST


# ── handle_llm: retry failure (outcome = "failed") ──────────────────


def test_handle_llm_records_failed_outcome_when_retry_raises(
    llm_engine: Engine,
    test_provider: TestProvider,
) -> None:
    """When a retry also fails, the incident outcome is set to `failed`
    and the exception propagates to the caller.
    """

    async def _always_fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise TypeError("'NoneType' object is not subscriptable")
        yield ""  # make it a generator

    test_provider.set_model(FunctionModel(stream_function=_always_fail))

    with pytest.raises(TypeError, match="NoneType"):
        handle_llm(llm_engine._llm_context(), "hello")

    # Incident: outcome is "failed", not "recovered".
    incidents = _failure_incidents(llm_engine)
    assert len(incidents) == 1
    assert incidents[0].failure_kind == FailureKind.TYPE_ERROR
    assert incidents[0].outcome == IncidentOutcome.FAILED


# ── _prepare_turn: dangling tool repair is transient ─────────────────


def test_dangling_tool_repair_is_transient(
    llm_engine: Engine,
    test_provider: TestProvider,
) -> None:
    """`repair_dangling_tool_calls` must not persist synthetic messages.

    The in-memory history passed to the agent contains the synthetic
    `(cancelled)` tool returns, but the DB retains the original
    dangling state.  An `LLM_HISTORY_REPAIR` incident row is persisted.
    """

    # Seed history with a dangling tool call (cancelled mid-turn).
    dangling_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="read my files")]),
        ModelResponse(
            parts=[
                TextPart(content="Let me check…"),
                ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
            ],
            model_name="test",
        ),
        # No ToolReturnPart — simulates Ctrl+C mid-tool-call.
    ]
    llm_engine.store.save_messages(llm_engine.state.session_id, dangling_history)

    handle_llm(llm_engine._llm_context(), "continue")

    # The original dangling history is untouched in the DB.
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    has_synthetic = any(
        isinstance(msg, ModelRequest)
        and any(isinstance(p, ToolReturnPart) and p.content == "(cancelled)" for p in msg.parts)
        for msg in loaded
    )
    assert not has_synthetic, "Synthetic (cancelled) messages must not be persisted"

    # Incident: LLM_HISTORY_REPAIR row for REPAIR_DANGLING.
    repairs = _repair_incidents(llm_engine)
    dangling_repairs = [r for r in repairs if r.strategy == RecoveryStrategy.REPAIR_DANGLING]
    assert len(dangling_repairs) == 1
    assert dangling_repairs[0].tool_names == ["read_file"]
    assert dangling_repairs[0].call_count == 1
    assert dangling_repairs[0].reason == "cancelled_mid_tool_call"


# ── _prepare_turn: simplify_history persists incidents ───────────────────


def test_simplify_history_persists_incidents(
    llm_engine: Engine,
    test_provider: TestProvider,
) -> None:
    """When `handle_llm` retries with simplified history, it persists
    `LLM_HISTORY_REPAIR` rows for `demote_thinking` and
    `flatten_tool_exchanges` with correct counts.
    """

    # Seed history with a mixed request (tool return + user prompt)
    # that consolidation will split, plus thinking parts.
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="review this")]),
        ModelResponse(
            parts=[
                ThinkingPart(content="Let me reason…", id="rs_1"),
                TextPart(content="I'll check the files."),
                ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
            ],
            model_name="test",
        ),
        # Mixed: tool return + user prompt triggers consolidation.
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="file contents", tool_call_id="tc1"),
                UserPromptPart(content="also check tests"),
            ]
        ),
        ModelResponse(
            parts=[
                ThinkingPart(content="Now I understand…", id="rs_2"),
                TextPart(content="Here are my findings."),
            ],
            model_name="test",
        ),
    ]
    llm_engine.store.save_messages(llm_engine.state.session_id, history)

    # First model call raises history-format error.
    # Retry with simplified history succeeds.
    call_count = 0

    async def _format_error_then_ok(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_http_error(
                HTTPStatus.BAD_REQUEST,
                "tool_use ids ['tc1'] without matching tool_result blocks",
            )
        yield "fixed"

    test_provider.set_model(FunctionModel(stream_function=_format_error_then_ok))

    handle_llm(llm_engine._llm_context(), "now fix it")

    # Incident: LLM_ATTEMPT_FAILED for the history-format error.
    failures = _failure_incidents(llm_engine)
    assert len(failures) == 1
    assert failures[0].failure_kind == FailureKind.HISTORY_FORMAT
    assert failures[0].outcome == IncidentOutcome.RECOVERED

    # Incident: LLM_HISTORY_REPAIR for consolidation (level 1 retry).
    repairs = _repair_incidents(llm_engine)
    consolidate = [r for r in repairs if r.strategy == RecoveryStrategy.CONSOLIDATE_TOOL_RETURNS]
    assert len(consolidate) == 1
    assert consolidate[0].turns_fixed == 1
