"""Tests that streamed TextDelta events match finalized message content.

The key invariant: text accumulated from all `TextDelta` events
(what the UI displays) must equal the `TextPart.content` in the
finalized `ModelResponse` (what the DB stores).

Tests go through `engine.run_task` with `FunctionModel` via the
test provider so the full pipeline runs end-to-end.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from rbtr.engine.core import Engine
from rbtr.engine.types import TaskType
from rbtr.events import Event, TextDelta, ToolCallFinished, ToolCallStarted
from tests.helpers import StubProvider, drain

# ── Helpers ──────────────────────────────────────────────────────────


def _collect_text(events: list[Event]) -> str:
    """Concatenate all `TextDelta` payloads from drained events."""
    return "".join(e.delta for e in events if isinstance(e, TextDelta))


def _final_text(engine: Engine) -> str:
    """Extract concatenated text from all `TextPart`s in DB responses."""
    parts: list[str] = []
    for msg in engine.store.load_messages(engine.state.session_id):
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    parts.append(part.content)
    return "".join(parts)


# ── Text fidelity tests ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["Hello world!"], "Hello world!"),
        (["Ass", "umption: you want"], "Assumption: you want"),
        (["A", "B", "C", "D", "E"], "ABCDE"),
        (["x" * 5000], "x" * 5000),
        (["Hello ", "🌍", " — done"], "Hello 🌍 — done"),
    ],
    ids=[
        "single_chunk",
        "multi_chunk",
        "many_small_chunks",
        "one_large_chunk",
        "unicode",
    ],
)
def test_streamed_text_matches_final_message(
    llm_engine: Engine,
    stub_provider: StubProvider,
    chunks: list[str],
    expected: str,
) -> None:
    """Accumulated `TextDelta` events equal the finalized `TextPart.content`."""

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    stub_provider.set_model(FunctionModel(stream_function=stream_fn))
    llm_engine.run_task(TaskType.LLM, "test")
    events = drain(llm_engine.events)

    streamed = _collect_text(events)
    final = _final_text(llm_engine)

    assert streamed == expected
    assert final == expected
    assert streamed == final


def test_text_after_tool_calls_matches(
    llm_engine: Engine,
    stub_provider: StubProvider,
) -> None:
    """Text emitted after a tool-call round-trip is fully captured."""
    call_count = 0

    async def stream_fn(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {0: DeltaToolCall(name="read_file", json_args='{"path": "a.py"}')}
        else:
            yield "Here"
            yield " are the"
            yield " results."

    stub_provider.set_model(FunctionModel(stream_function=stream_fn))
    llm_engine.run_task(TaskType.LLM, "read a.py")
    events = drain(llm_engine.events)

    tool_starts = [e for e in events if isinstance(e, ToolCallStarted)]
    tool_ends = [e for e in events if isinstance(e, ToolCallFinished)]
    assert len(tool_starts) == 1
    assert len(tool_ends) == 1

    streamed = _collect_text(events)
    final = _final_text(llm_engine)

    assert streamed == "Here are the results."
    assert streamed == final


def test_text_before_and_after_tool_call(
    llm_engine: Engine,
    stub_provider: StubProvider,
) -> None:
    """Text before and after a tool call is fully captured."""
    call_count = 0

    async def stream_fn(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield "Let me check"
            yield {0: DeltaToolCall(name="read_file", json_args='{"path": "b.py"}')}
        else:
            yield "Found it: OK"

    stub_provider.set_model(FunctionModel(stream_function=stream_fn))
    llm_engine.run_task(TaskType.LLM, "check b.py")
    events = drain(llm_engine.events)

    streamed = _collect_text(events)
    final = _final_text(llm_engine)

    assert "Let me check" in streamed
    assert "Found it: OK" in streamed
    assert streamed == final


def test_db_content_matches_streamed_text(
    llm_engine: Engine,
    stub_provider: StubProvider,
) -> None:
    """Text persisted to the DB matches what was streamed to the UI."""

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield "Database"
        yield " and UI"
        yield " must match."

    stub_provider.set_model(FunctionModel(stream_function=stream_fn))
    llm_engine.run_task(TaskType.LLM, "test")
    events = drain(llm_engine.events)

    streamed = _collect_text(events)
    final = _final_text(llm_engine)

    assert streamed == "Database and UI must match."
    assert streamed == final
