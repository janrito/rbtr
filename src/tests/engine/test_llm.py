"""Tests for engine/llm.py — event emission, usage tracking, error
classification, and overflow/retry behaviour.

Tests target observable outcomes (events emitted, state mutated,
handlers retried) rather than internal async iteration mechanics.
"""

from __future__ import annotations

from pathlib import Path
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
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
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
    engine: Engine, tool_name: str, args: str | dict[str, str] | None, expected_args: str
) -> None:
    """Tool call events are emitted with correctly formatted args."""
    _emit_tool_event(engine, _make_tool_call_event(tool_name, args))
    events = drain(engine.events)
    assert len(events) == 1
    assert isinstance(events[0], ToolCallStarted)
    assert events[0].tool_name == tool_name
    assert expected_args in events[0].args


def test_emit_tool_result_normal(engine: Engine) -> None:
    """Normal tool result emits content text."""
    _emit_tool_event(engine, _make_tool_result_event("read_file", "file contents here"))
    events = drain(engine.events)
    assert len(events) == 1
    assert isinstance(events[0], ToolCallFinished)
    assert events[0].result == "file contents here"


def test_emit_tool_result_truncation(engine: Engine) -> None:
    """Long results are truncated at tool_max_chars."""
    from rbtr.config import config

    long_content = "x" * (config.tui.tool_max_chars + 100)
    _emit_tool_event(engine, _make_tool_result_event("read_file", long_content))
    events = drain(engine.events)
    assert len(events[0].result) == config.tui.tool_max_chars + 1  # +1 for "…"
    assert events[0].result.endswith("…")


def test_emit_tool_result_retry(engine: Engine) -> None:
    """Retry prompt result emits '(retry)'."""
    _emit_tool_event(engine, _make_tool_retry_event("read_file", "tool failed, try again"))
    events = drain(engine.events)
    assert isinstance(events[0], ToolCallFinished)
    assert events[0].result == "(retry)"


# ── _record_usage ────────────────────────────────────────────────────


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
def test_record_usage_basics(
    engine: Engine,
    messages: list[ModelMessage],
    input_tokens: int,
    output_tokens: int,
    expected_cost: float | None,
    expected_input: int,
    expected_output: int,
) -> None:
    """Basic _record_usage scenarios: empty, with response, no model name."""
    _record_usage(
        engine, messages, _make_run_usage(input_tokens=input_tokens, output_tokens=output_tokens)
    )
    if expected_cost is not None:
        assert engine.state.usage.total_cost == expected_cost
    assert engine.state.usage.input_tokens == expected_input
    assert engine.state.usage.output_tokens == expected_output


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

    _record_usage(engine, messages, _make_run_usage(input_tokens=5000, output_tokens=200))

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
    _record_usage(engine, messages, _make_run_usage(input_tokens=13000, output_tokens=400))

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

    _record_usage(engine, messages, _make_run_usage(input_tokens=100_000))

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

    _record_usage(engine, messages, usage)

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
    _record_usage(engine, msgs1, _make_run_usage(input_tokens=10_000))

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
    _record_usage(engine, new_msgs2, _make_run_usage(input_tokens=50_000))

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


def test_auto_compact_on_overflow_compacts_and_retries(
    mocker: MockerFixture,
    config_path: Path,
    creds_path: Path,
    engine: Engine,
) -> None:
    """Overflow handler compacts history then retries via handle_llm."""

    creds.update(openai_api_key="sk-test")
    engine.state.openai_connected = True
    engine.state.model_name = "openai/gpt-4o"

    compact_mock = mocker.patch("rbtr.engine.compact.compact_history")

    retry_calls: list[str] = []
    mocker.patch("rbtr.engine.llm.handle_llm", lambda eng, msg: retry_calls.append(msg))

    _auto_compact_on_overflow(engine, "my question")

    compact_mock.assert_called_once()
    assert retry_calls == ["my question"]


# ── handle_llm context overflow integration ──────────────────────────


def test_handle_llm_context_overflow_triggers_compact(
    mocker: MockerFixture,
    config_path: Path,
    creds_path: Path,
    engine: Engine,
) -> None:
    """handle_llm auto-compacts on context overflow and retries."""

    creds.update(openai_api_key="sk-test")
    engine.state.openai_connected = True
    engine.state.model_name = "openai/gpt-4o"
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="analyse the codebase")]),
        ModelResponse(parts=[TextPart(content="I found 3 files so far")], model_name="test"),
        ModelRequest(parts=[UserPromptPart(content="turn 2")]),
        ModelResponse(parts=[TextPart(content="resp 2")], model_name="test"),
    ]
    engine._sync_store_context()
    engine.store.save_messages(engine.state.session_id, messages)

    # First call to _run_agent raises overflow, second succeeds
    call_count = 0

    def fake_run_agent(eng: Engine, model: object, msg: str, **kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_http_error(400, "maximum context length exceeded")

    mocker.patch("rbtr.engine.llm._run_agent", fake_run_agent)

    # Mock compact_history (a no-op here — the retry is what matters).
    mocker.patch("rbtr.engine.compact.compact_history")

    from rbtr.engine.llm import handle_llm

    handle_llm(engine, "test question")

    # _run_agent called twice: first fails, compact, then retry via handle_llm
    # But handle_llm calls _run_agent for the retry, so total = 2
    assert call_count == 2
