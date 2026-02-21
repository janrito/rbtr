"""Tests for engine/llm.py helper functions.

Covers ``_emit_tool_event``, ``_record_usage``, and the
``UsageLimitExceeded`` → summary flow in ``_stream_agent``.
"""

from __future__ import annotations

import asyncio

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage

from rbtr.engine.llm import _emit_tool_event, _record_usage
from rbtr.events import ToolCallFinished, ToolCallStarted

from .conftest import drain, make_engine

# ── Shared data ──────────────────────────────────────────────────────

# A minimal FunctionToolCallEvent-like object. We construct real
# pydantic-ai event objects via their public constructors.


def _make_tool_call_event(tool_name: str, args):
    """Build a FunctionToolCallEvent with the given args."""
    from pydantic_ai.messages import FunctionToolCallEvent, ToolCallPart

    part = ToolCallPart(
        tool_name=tool_name,
        args=args,
        tool_call_id="call_1",
    )
    return FunctionToolCallEvent(part=part)


def _make_tool_result_event(tool_name: str, content: str):
    """Build a FunctionToolResultEvent with a ToolReturnPart."""
    from pydantic_ai.messages import FunctionToolResultEvent, ToolReturnPart

    result = ToolReturnPart(
        tool_name=tool_name,
        content=content,
        tool_call_id="call_1",
    )
    return FunctionToolResultEvent(result=result)


def _make_tool_retry_event(tool_name: str, content: str):
    """Build a FunctionToolResultEvent with a RetryPromptPart (retry)."""
    from pydantic_ai.messages import FunctionToolResultEvent, RetryPromptPart

    result = RetryPromptPart(
        tool_name=tool_name,
        content=content,
        tool_call_id="call_1",
    )
    return FunctionToolResultEvent(result=result)


# ── _emit_tool_event: tool calls ─────────────────────────────────────


def test_emit_tool_call_dict_args() -> None:
    """Dict args are serialised as JSON."""
    engine, events, _ = make_engine()
    event = _make_tool_call_event("read_file", {"path": "foo.py"})

    _emit_tool_event(engine, event)
    evts = drain(events)

    assert len(evts) == 1
    assert isinstance(evts[0], ToolCallStarted)
    assert evts[0].tool_name == "read_file"
    assert '"path"' in evts[0].args
    assert "foo.py" in evts[0].args


def test_emit_tool_call_non_dict_args() -> None:
    """Non-dict args use str()."""
    engine, events, _ = make_engine()
    event = _make_tool_call_event("search", "hello world")

    _emit_tool_event(engine, event)
    evts = drain(events)

    assert len(evts) == 1
    assert isinstance(evts[0], ToolCallStarted)
    assert evts[0].args == "hello world"


def test_emit_tool_call_none_args() -> None:
    """None args produce empty string."""
    engine, events, _ = make_engine()
    event = _make_tool_call_event("list_files", None)

    _emit_tool_event(engine, event)
    evts = drain(events)

    assert len(evts) == 1
    assert isinstance(evts[0], ToolCallStarted)
    assert evts[0].args == ""


# ── _emit_tool_event: tool results ──────────────────────────────────


def test_emit_tool_result_normal() -> None:
    """Normal tool result emits content text."""
    engine, events, _ = make_engine()
    event = _make_tool_result_event("read_file", "file contents here")

    _emit_tool_event(engine, event)
    evts = drain(events)

    assert len(evts) == 1
    assert isinstance(evts[0], ToolCallFinished)
    assert evts[0].tool_name == "read_file"
    assert evts[0].result == "file contents here"


def test_emit_tool_result_truncation() -> None:
    """Long results are truncated at tool_max_chars."""
    from rbtr.config import config

    engine, events, _ = make_engine()
    long_content = "x" * (config.tui.tool_max_chars + 100)
    event = _make_tool_result_event("read_file", long_content)

    _emit_tool_event(engine, event)
    evts = drain(events)

    assert len(evts) == 1
    result = evts[0].result
    assert len(result) == config.tui.tool_max_chars + 1  # +1 for "…"
    assert result.endswith("…")


def test_emit_tool_result_retry() -> None:
    """Retry prompt result emits '(retry)'."""
    engine, events, _ = make_engine()
    event = _make_tool_retry_event("read_file", "tool failed, try again")

    _emit_tool_event(engine, event)
    evts = drain(events)

    assert len(evts) == 1
    assert isinstance(evts[0], ToolCallFinished)
    assert evts[0].result == "(retry)"


# ── _record_usage ────────────────────────────────────────────────────


def _make_run_usage(*, input_tokens=100, output_tokens=50) -> RunUsage:
    return RunUsage(
        requests=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def test_record_usage_no_new_messages() -> None:
    """No new messages → usage tokens still recorded, zero cost."""
    engine, _, session = make_engine()
    history: list[ModelMessage] = []
    messages: list[ModelMessage] = []
    usage = _make_run_usage()

    _record_usage(engine, history, messages, usage)

    assert session.usage.total_cost == 0.0


def test_record_usage_with_model_response() -> None:
    """ModelResponse with model_name records usage."""
    engine, _, session = make_engine()
    history: list[ModelMessage] = []
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[TextPart(content="hello")],
            model_name="claude-sonnet-4-20250514",
        ),
    ]
    usage = _make_run_usage(input_tokens=500, output_tokens=200)

    _record_usage(engine, history, messages, usage)

    assert session.usage.input_tokens == 500
    assert session.usage.output_tokens == 200


def test_record_usage_skips_messages_without_model_name() -> None:
    """ModelResponse without model_name is skipped for cost."""
    engine, _, session = make_engine()
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
    assert session.usage.total_cost == 0.0


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


def test_stream_agent_catches_limit_and_triggers_summary(mocker, creds_path) -> None:
    """When UsageLimitExceeded fires, messages are preserved and _stream_summary is called."""
    from pydantic_ai.exceptions import UsageLimitExceeded

    from rbtr.creds import creds
    from rbtr.engine.llm import _stream_agent

    creds.update(openai_api_key="sk-test")
    engine, _events, session = make_engine()
    session.openai_connected = True
    session.model_name = "openai/gpt-4o"

    partial_messages: list[ModelMessage] = [_USER_MSG, _PARTIAL_RESPONSE]

    # Mock agent.iter to raise UsageLimitExceeded after yielding partial messages
    class _FakeRun:
        def all_messages(self):
            return partial_messages

        def usage(self):
            return RunUsage(requests=25, input_tokens=1000, output_tokens=500)

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise UsageLimitExceeded("request_limit of 25")

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

    # Mock _stream_summary so we can verify it's called
    summary_called = False
    original_settings = [None]

    async def fake_summary(eng, model, settings):
        nonlocal summary_called
        summary_called = True
        original_settings[0] = settings
        # Simulate what _stream_summary does: append summary to history
        eng.session.message_history = [*eng.session.message_history, _SUMMARY_RESPONSE]

    mocker.patch("rbtr.engine.llm._stream_summary", fake_summary)

    # Need an event loop for _stream_agent
    from rbtr.providers import build_model

    model = build_model("openai/gpt-4o")
    asyncio.run(_stream_agent(engine, model, "analyse the codebase"))

    # Partial messages preserved in history
    assert _USER_MSG in session.message_history
    assert _PARTIAL_RESPONSE in session.message_history
    # Summary was appended
    assert _SUMMARY_RESPONSE in session.message_history
    assert summary_called


def test_stream_summary_sends_prompt_and_appends_history(mocker, creds_path) -> None:
    """_stream_summary uses a tool-free agent, passes the limit prompt, and appends to history."""
    from rbtr.creds import creds
    from rbtr.engine.llm import _LIMIT_SUMMARY_PROMPT, _stream_summary

    creds.update(openai_api_key="sk-test")
    engine, _events, session = make_engine()
    session.openai_connected = True
    session.model_name = "openai/gpt-4o"
    session.message_history = [_USER_MSG, _PARTIAL_RESPONSE]

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
    assert _USER_MSG in session.message_history
    assert _PARTIAL_RESPONSE in session.message_history
    assert summary_msg in session.message_history


def test_stream_agent_no_limit_hit_skips_summary(mocker, creds_path) -> None:
    """Normal completion (no limit hit) does not trigger _stream_summary."""
    from rbtr.creds import creds
    from rbtr.engine.llm import _stream_agent

    creds.update(openai_api_key="sk-test")
    engine, _events, session = make_engine()
    session.openai_connected = True
    session.model_name = "openai/gpt-4o"

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
    assert len(session.message_history) == 2
