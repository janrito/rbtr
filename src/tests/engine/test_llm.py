"""Tests for engine/llm.py helper functions.

Covers ``_emit_tool_event``, ``_record_usage``, ``_update_live_usage``,
``_is_context_overflow``, ``_auto_compact_on_overflow``,
and the ``UsageLimitExceeded`` → summary flow.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
from pytest_mock import MockerFixture

from rbtr.creds import creds
from rbtr.engine.core import Engine
from rbtr.engine.llm import (
    _auto_compact_on_overflow,
    _emit_tool_event,
    _is_context_overflow,
    _is_effort_unsupported,
    _record_usage,
    _stream_agent,
    _update_live_usage,
    resolve_model_settings,
)
from rbtr.events import ToolCallFinished, ToolCallStarted

from .conftest import drain

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


# ── _emit_tool_event: tool calls ─────────────────────────────────────


def test_emit_tool_call_dict_args(engine: Engine) -> None:
    """Dict args are serialised as JSON."""
    event = _make_tool_call_event("read_file", {"path": "foo.py"})

    _emit_tool_event(engine, event)
    drained_events = drain(engine.events)

    assert len(drained_events) == 1
    assert isinstance(drained_events[0], ToolCallStarted)
    assert drained_events[0].tool_name == "read_file"
    assert '"path"' in drained_events[0].args
    assert "foo.py" in drained_events[0].args


def test_emit_tool_call_non_dict_args(engine: Engine) -> None:
    """Non-dict args use str()."""
    event = _make_tool_call_event("search", "hello world")

    _emit_tool_event(engine, event)
    drained_events = drain(engine.events)

    assert len(drained_events) == 1
    assert isinstance(drained_events[0], ToolCallStarted)
    assert drained_events[0].args == "hello world"


def test_emit_tool_call_none_args(engine: Engine) -> None:
    """None args produce empty string."""
    event = _make_tool_call_event("list_files", None)

    _emit_tool_event(engine, event)
    drained_events = drain(engine.events)

    assert len(drained_events) == 1
    assert isinstance(drained_events[0], ToolCallStarted)
    assert drained_events[0].args == ""


# ── _emit_tool_event: tool results ──────────────────────────────────


def test_emit_tool_result_normal(engine: Engine) -> None:
    """Normal tool result emits content text."""
    event = _make_tool_result_event("read_file", "file contents here")

    _emit_tool_event(engine, event)
    drained_events = drain(engine.events)

    assert len(drained_events) == 1
    assert isinstance(drained_events[0], ToolCallFinished)
    assert drained_events[0].tool_name == "read_file"
    assert drained_events[0].result == "file contents here"


def test_emit_tool_result_truncation(engine: Engine) -> None:
    """Long results are truncated at tool_max_chars."""
    from rbtr.config import config

    long_content = "x" * (config.tui.tool_max_chars + 100)
    event = _make_tool_result_event("read_file", long_content)

    _emit_tool_event(engine, event)
    drained_events = drain(engine.events)

    assert len(drained_events) == 1
    result = drained_events[0].result
    assert len(result) == config.tui.tool_max_chars + 1  # +1 for "…"
    assert result.endswith("…")


def test_emit_tool_result_retry(engine: Engine) -> None:
    """Retry prompt result emits '(retry)'."""
    event = _make_tool_retry_event("read_file", "tool failed, try again")

    _emit_tool_event(engine, event)
    drained_events = drain(engine.events)

    assert len(drained_events) == 1
    assert isinstance(drained_events[0], ToolCallFinished)
    assert drained_events[0].result == "(retry)"


# ── _record_usage ────────────────────────────────────────────────────


def _make_run_usage(*, input_tokens: int = 100, output_tokens: int = 50) -> RunUsage:
    return RunUsage(
        requests=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def test_record_usage_no_new_messages(engine: Engine) -> None:
    """No new messages → usage tokens still recorded, zero cost."""
    history: list[ModelMessage] = []
    messages: list[ModelMessage] = []
    usage = _make_run_usage()

    _record_usage(engine, history, messages, usage)

    assert engine.state.usage.total_cost == 0.0


def test_record_usage_with_model_response(engine: Engine) -> None:
    """ModelResponse with model_name records usage."""
    history: list[ModelMessage] = []
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[TextPart(content="hello")],
            model_name="claude-sonnet-4-20250514",
        ),
    ]
    usage = _make_run_usage(input_tokens=500, output_tokens=200)

    _record_usage(engine, history, messages, usage)

    assert engine.state.usage.input_tokens == 500
    assert engine.state.usage.output_tokens == 200


def test_record_usage_skips_messages_without_model_name(engine: Engine) -> None:
    """ModelResponse without model_name is skipped for cost."""
    history: list[ModelMessage] = []
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[TextPart(content="hello")],
            model_name=None,
        ),
    ]
    usage = _make_run_usage()

    _record_usage(engine, history, messages, usage)

    # Tokens are still recorded from RunUsage, but cost stays 0
    assert engine.state.usage.total_cost == 0.0


# ── _record_usage: context tracking ─────────────────────────────────

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
    from pydantic_ai.usage import RequestUsage

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
def test_record_usage_sets_context_window(
    model_name: str,
    provider: str,
    expected_window: int,
    engine: Engine,
) -> None:
    """_record_usage resolves the real context window for built-in providers."""
    engine.state.model_name = model_name
    response = _make_provider_response(provider, input_tokens=5000, output_tokens=200)
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        response,
    ]

    _record_usage(engine, [], messages, _make_run_usage(input_tokens=5000, output_tokens=200))

    assert engine.state.usage.context_window == expected_window
    assert engine.state.usage.context_window_known is True
    assert engine.state.usage.last_input_tokens == 5000


def test_record_usage_last_input_tokens_from_response(engine: Engine) -> None:
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
    _record_usage(engine, [], messages, _make_run_usage(input_tokens=13000, output_tokens=400))

    # last_input_tokens should be the last response's value, not cumulative
    assert engine.state.usage.last_input_tokens == 8000
    assert engine.state.usage.input_tokens == 13000


def test_record_usage_context_used_pct(engine: Engine) -> None:
    """context_used_pct uses the real context window, not the 128 k default."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    response = _make_provider_response(input_tokens=100_000)
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        response,
    ]

    _record_usage(engine, [], messages, _make_run_usage(input_tokens=100_000))

    # 100k / 200k = 50%, NOT 100k / 128k = 78%
    assert engine.state.usage.context_window == 200_000
    assert engine.state.usage.context_used_pct == pytest.approx(50.0)


def test_record_usage_cache_tokens(engine: Engine) -> None:
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

    _record_usage(engine, [], messages, usage)

    assert engine.state.usage.cache_read_tokens == 3000
    assert engine.state.usage.cache_write_tokens == 1000
    assert engine.state.usage.context_window == 200_000


def test_record_usage_multi_turn_context_pct(engine: Engine) -> None:
    """Context % stays correct across multiple turns."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"

    # Turn 1
    resp1 = _make_provider_response(input_tokens=10_000)
    msgs1: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="turn 1")]),
        resp1,
    ]
    _record_usage(engine, [], msgs1, _make_run_usage(input_tokens=10_000))

    assert engine.state.usage.context_window == 200_000
    assert engine.state.usage.context_used_pct == pytest.approx(5.0)

    # Turn 2 — context grows
    resp2 = _make_provider_response(input_tokens=50_000)
    msgs2: list[ModelMessage] = [
        *msgs1,
        ModelRequest(parts=[UserPromptPart(content="turn 2")]),
        resp2,
    ]
    _record_usage(engine, msgs1, msgs2, _make_run_usage(input_tokens=50_000))

    assert engine.state.usage.context_window == 200_000  # stays correct
    assert engine.state.usage.context_used_pct == pytest.approx(25.0)


# ── _update_live_usage: mid-run context tracking ─────────────────────


def test_update_live_usage_sets_context_window(engine: Engine) -> None:
    """_update_live_usage resolves the context window during streaming."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine.state.usage.snapshot_base()

    response = _make_provider_response(input_tokens=10_000, output_tokens=500)
    run_usage = RunUsage(requests=1, input_tokens=10_000, output_tokens=500)

    _update_live_usage(engine, run_usage, response)

    assert engine.state.usage.last_input_tokens == 10_000
    assert engine.state.usage.context_window == 200_000
    assert engine.state.usage.context_window_known is True
    # 10k / 200k = 5%, NOT 10k / 128k = 7.8%
    assert engine.state.usage.context_used_pct == pytest.approx(5.0)


def test_update_live_usage_accumulates_over_base(engine: Engine) -> None:
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

    _update_live_usage(engine, run_usage, response)

    # Lifetime totals: 5000 + 8000 = 13000
    assert engine.state.usage.input_tokens == 13_000
    assert engine.state.usage.output_tokens == 500
    # But context % uses only last request's tokens
    assert engine.state.usage.last_input_tokens == 8000
    assert engine.state.usage.context_used_pct == pytest.approx(4.0)


# ── UsageLimitExceeded → summary flow ────────────────────────────────

# Pre-built messages for the limit-hit scenario.
_USER_MSG = ModelRequest(parts=[UserPromptPart(content="analyse the codebase")])
_PARTIAL_RESPONSE = ModelResponse(
    parts=[TextPart(content="I found 3 files so far")],
    model_name="test-model",
)
_SUMMARY_RESPONSE = ModelResponse(
    parts=[TextPart(content="I reviewed 3 files. 2 more remain.")],
    model_name="test-model",
)


def test_stream_agent_catches_limit_and_triggers_summary(
    mocker: MockerFixture,
    creds_path: Path,
    engine: Engine,
) -> None:
    """When UsageLimitExceeded fires, messages are preserved and _stream_summary is called."""

    creds.update(openai_api_key="sk-test")
    engine.state.openai_connected = True
    engine.state.model_name = "openai/gpt-4o"

    partial_messages: list[ModelMessage] = [_USER_MSG, _PARTIAL_RESPONSE]

    # Mock agent.iter to raise UsageLimitExceeded after yielding partial messages
    class _FakeRun:
        def all_messages(self) -> list[ModelRequest | ModelResponse]:
            return partial_messages

        def usage(self) -> RunUsage:
            return RunUsage(requests=25, input_tokens=1000, output_tokens=500)

        def __aiter__(self) -> _FakeRun:
            return self

        async def __anext__(self) -> None:
            raise UsageLimitExceeded("request_limit of 25")

        @property
        def ctx(self) -> None:
            return None

    class _FakeIterCtx:
        def __init__(self) -> None:
            self.run = _FakeRun()

        async def __aenter__(self) -> _FakeRun:
            return self.run

        async def __aexit__(self, *args: Any) -> None:
            pass

    mocker.patch("rbtr.engine.llm.agent.iter", return_value=_FakeIterCtx())

    # Mock _stream_summary so we can verify it's called
    summary_called = False
    original_settings = [None]

    async def fake_summary(eng: Engine, model: Model, settings: ModelSettings) -> None:
        nonlocal summary_called
        summary_called = True
        original_settings[0] = settings
        # Simulate what _stream_summary does: append summary to history
        eng.state.message_history = [*eng.state.message_history, _SUMMARY_RESPONSE]

    mocker.patch("rbtr.engine.llm._stream_summary", fake_summary)

    # Need an event loop for _stream_agent
    from rbtr.providers import build_model

    model = build_model("openai/gpt-4o")
    asyncio.run(_stream_agent(engine, model, "analyse the codebase"))

    # Partial messages preserved in history
    assert _USER_MSG in engine.state.message_history
    assert _PARTIAL_RESPONSE in engine.state.message_history
    # Summary was appended
    assert _SUMMARY_RESPONSE in engine.state.message_history
    assert summary_called


def test_stream_summary_sends_prompt_and_appends_history(
    mocker: MockerFixture,
    creds_path: Path,
    engine: Engine,
) -> None:
    """_stream_summary uses a tool-free agent, passes the limit prompt, and appends to history."""
    from rbtr.creds import creds
    from rbtr.engine.llm import _LIMIT_SUMMARY_PROMPT, _stream_summary

    creds.update(openai_api_key="sk-test")
    engine.state.openai_connected = True
    engine.state.model_name = "openai/gpt-4o"
    engine.state.message_history = [_USER_MSG, _PARTIAL_RESPONSE]

    # Track what Agent.iter receives
    captured_prompt: list[str | None] = [None]
    captured_history: list[list[ModelMessage] | None] = [None]

    summary_request = ModelRequest(parts=[UserPromptPart(content=_LIMIT_SUMMARY_PROMPT)])
    summary_msg = ModelResponse(
        parts=[TextPart(content="Summary: 3 files reviewed.")],
        model_name="test-model",
    )

    class _FakeRun:
        def __init__(self, history: list[ModelMessage] | None) -> None:
            self._history = history

        def all_messages(self) -> list[ModelMessage]:
            return [
                *list(self._history or []),
                summary_request,
                summary_msg,
            ]

        def __aiter__(self) -> _FakeRun:
            return self

        async def __anext__(self) -> None:
            # Yield no nodes — the test checks prompt/history, not streaming
            raise StopAsyncIteration

    class _FakeIterCtx:
        def __init__(self, prompt: str, history: list[ModelMessage] | None) -> None:
            captured_prompt[0] = prompt
            captured_history[0] = history
            self.run = _FakeRun(history)

        async def __aenter__(self) -> _FakeRun:
            return self.run

        async def __aexit__(self, *args: object) -> None:
            pass

    class FakeAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def iter(
            self,
            prompt: str,
            *,
            model: object = None,
            message_history: list[ModelMessage] | None = None,
            model_settings: object = None,
            usage_limits: object = None,
        ) -> _FakeIterCtx:
            return _FakeIterCtx(prompt, message_history)

    mocker.patch("rbtr.engine.llm.Agent", FakeAgent)

    from rbtr.providers import build_model

    model = build_model("openai/gpt-4o")
    asyncio.run(_stream_summary(engine, model, None))

    # Verify the prompt sent to the summary agent
    assert captured_prompt[0] == _LIMIT_SUMMARY_PROMPT
    # Verify conversation history was forwarded
    assert captured_history[0] is not None
    assert len(captured_history[0]) == 2  # _USER_MSG + _PARTIAL_RESPONSE

    # Verify summary messages appended — original history preserved
    assert _USER_MSG in engine.state.message_history
    assert _PARTIAL_RESPONSE in engine.state.message_history
    assert summary_msg in engine.state.message_history


def test_stream_agent_no_limit_hit_skips_summary(
    mocker: MockerFixture, creds_path: Path, engine: Engine
) -> None:
    """Normal completion (no limit hit) does not trigger _stream_summary."""
    from rbtr.creds import creds
    from rbtr.engine.llm import _stream_agent

    creds.update(openai_api_key="sk-test")
    engine.state.openai_connected = True
    engine.state.model_name = "openai/gpt-4o"

    completed_messages: list[ModelMessage] = [
        _USER_MSG,
        ModelResponse(parts=[TextPart(content="Done!")], model_name="test"),
    ]

    class _FakeRun:
        def all_messages(self):
            return completed_messages

        def usage(self):
            return RunUsage(requests=5, input_tokens=500, output_tokens=200)

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        @property
        def ctx(self):
            return None

    class _FakeIterCtx:
        def __init__(self):
            self.run = _FakeRun()

        async def __aenter__(self):
            return self.run

        async def __aexit__(self, *args):
            pass

    mocker.patch("rbtr.engine.llm.agent.iter", return_value=_FakeIterCtx())

    summary_called = False

    async def fake_summary(eng, model, settings):
        nonlocal summary_called
        summary_called = True

    mocker.patch("rbtr.engine.llm._stream_summary", fake_summary)

    from rbtr.providers import build_model

    model = build_model("openai/gpt-4o")
    asyncio.run(_stream_agent(engine, model, "hello"))

    assert not summary_called
    assert len(engine.state.message_history) == 2


# ── _is_context_overflow ─────────────────────────────────────────────


def _make_http_error(status: int, body: str) -> ModelHTTPError:
    """Construct a ModelHTTPError with given status and body text."""
    return ModelHTTPError(status, "test-model", body)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, "This model's maximum context length is 128000 tokens"),
        (400, "prompt is too long: 150000 tokens > 128000 token limit"),
        (400, "Request too large for model"),
        (400, "input is too long (150000 tokens, max 128000)"),
        (400, "content_too_large: message exceeds context window"),
        (400, "too many tokens in the prompt"),
        (413, "Payload too large"),
    ],
)
def test_is_context_overflow_positive(status: int, body: str) -> None:
    """Errors that indicate context overflow are detected."""
    exc = _make_http_error(status, body)
    assert _is_context_overflow(exc)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, "Invalid API key"),
        (400, "malformed request body"),
        (401, "Unauthorized"),
        (429, "Rate limit exceeded"),
        (500, "Internal server error"),
    ],
)
def test_is_context_overflow_negative(status: int, body: str) -> None:
    """Non-context errors are not misidentified."""
    exc = _make_http_error(status, body)
    assert not _is_context_overflow(exc)


# ── _is_effort_unsupported ────────────────────────────────────────────


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
    exc = _make_http_error(400, body)
    assert _is_effort_unsupported(exc)


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
    exc = _make_http_error(400, body)
    assert not _is_effort_unsupported(exc)


# ── resolve_model_settings: effort_supported ─────────────────────────


def test_resolve_model_settings_skips_effort_when_unsupported(mocker: MockerFixture) -> None:
    """When effort_supported=False, effort settings are omitted."""
    from pydantic_ai.models.anthropic import AnthropicModel

    mock_model = mocker.MagicMock(spec=AnthropicModel)

    settings = resolve_model_settings(mock_model, "claude/claude-sonnet-4-5-20250929")
    # Default effort is MEDIUM → should produce settings
    assert settings is not None

    # With effort_supported=False → no effort settings
    settings_no = resolve_model_settings(
        mock_model, "claude/claude-sonnet-4-5-20250929", effort_supported=False
    )
    assert settings_no is None


# ── handle_llm: effort-unsupported retry ─────────────────────────────


def test_handle_llm_retries_without_effort_on_rejection(
    mocker: MockerFixture,
    config_path: Path,
    creds_path: Path,
    engine: Engine,
) -> None:
    """handle_llm retries without effort when the model rejects it."""
    from rbtr.creds import creds
    from rbtr.engine.llm import handle_llm

    creds.update(openai_api_key="sk-test")
    engine.state.openai_connected = True
    engine.state.model_name = "openai/gpt-4o"

    call_count = 0

    def fake_run_agent(eng, model, msg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_http_error(400, "This model does not support the effort parameter.")

    mocker.patch("rbtr.engine.llm._run_agent", fake_run_agent)

    handle_llm(engine, "test question")

    assert call_count == 2
    assert engine.state.effort_supported is False


# ── _auto_compact_on_overflow ────────────────────────────────────────


def test_auto_compact_on_overflow_returns_false_short_history(engine: Engine) -> None:
    """No compaction when history has < 2 messages."""
    engine.state.message_history = [_USER_MSG]

    result = _auto_compact_on_overflow(engine, "test")
    assert result is False


def test_auto_compact_on_overflow_compacts_and_retries(
    mocker: MockerFixture,
    config_path: Path,
    creds_path: Path,
    engine: Engine,
) -> None:
    """When compaction reduces history, the message is retried."""
    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")
    engine.state.openai_connected = True
    engine.state.model_name = "openai/gpt-4o"

    # Build a history long enough to compact
    engine.state.message_history = [
        _USER_MSG,
        _PARTIAL_RESPONSE,
        ModelRequest(parts=[UserPromptPart(content="turn 2")]),
        ModelResponse(parts=[TextPart(content="resp 2")], model_name="test"),
        ModelRequest(parts=[UserPromptPart(content="turn 3")]),
        ModelResponse(parts=[TextPart(content="resp 3")], model_name="test"),
    ]

    # Mock compact_history to simulate reducing history
    def fake_compact(eng, extra_instructions=""):
        eng.state.message_history = eng.state.message_history[-2:]

    mocker.patch("rbtr.engine.compact.compact_history", fake_compact)

    # Mock handle_llm (the recursive call) to track it was called
    retry_calls: list[str] = []

    def fake_handle_llm(eng, msg):
        retry_calls.append(msg)

    mocker.patch("rbtr.engine.llm.handle_llm", fake_handle_llm)

    result = _auto_compact_on_overflow(engine, "my question")

    assert result is True
    assert retry_calls == ["my question"]


def test_auto_compact_on_overflow_no_reduction_returns_false(
    mocker: MockerFixture,
    config_path: Path,
    creds_path: Path,
    engine: Engine,
) -> None:
    """When compaction doesn't reduce history length, returns False."""
    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")
    engine.state.openai_connected = True
    engine.state.model_name = "openai/gpt-4o"

    engine.state.message_history = [
        _USER_MSG,
        _PARTIAL_RESPONSE,
    ]

    # Mock compact_history that does nothing (e.g. compaction failed)
    def fake_noop_compact(eng, extra_instructions=""):
        pass

    mocker.patch("rbtr.engine.compact.compact_history", fake_noop_compact)

    result = _auto_compact_on_overflow(engine, "my question")
    assert result is False


# ── handle_llm context overflow integration ──────────────────────────


def test_handle_llm_context_overflow_triggers_compact(
    mocker: MockerFixture,
    config_path: Path,
    creds_path: Path,
    engine: Engine,
) -> None:
    """handle_llm auto-compacts on context overflow and retries."""
    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")
    engine.state.openai_connected = True
    engine.state.model_name = "openai/gpt-4o"
    engine.state.message_history = [
        _USER_MSG,
        _PARTIAL_RESPONSE,
        ModelRequest(parts=[UserPromptPart(content="turn 2")]),
        ModelResponse(parts=[TextPart(content="resp 2")], model_name="test"),
    ]

    # First call to _run_agent raises overflow, second succeeds
    call_count = 0

    def fake_run_agent(eng, model, msg):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_http_error(400, "maximum context length exceeded")

    mocker.patch("rbtr.engine.llm._run_agent", fake_run_agent)

    # Mock compact_history to reduce history
    def fake_compact(eng, extra_instructions=""):
        eng.state.message_history = eng.state.message_history[-2:]

    mocker.patch("rbtr.engine.compact.compact_history", fake_compact)

    from rbtr.engine.llm import handle_llm

    handle_llm(engine, "test question")

    # _run_agent called twice: first fails, compact, then retry via handle_llm
    # But handle_llm calls _run_agent for the retry, so total = 2
    assert call_count == 2
