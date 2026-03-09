"""Tests that streamed TextDelta events match finalized message content.

The key invariant: text accumulated from all ``TextDelta`` events
(what the UI displays) must equal the ``TextPart.content`` in the
finalized ``ModelResponse`` (what the DB stores).

Tests use ``FunctionModel`` with controlled ``stream_function``
implementations so the real ``_do_stream`` pipeline runs end-to-end
— pydantic-ai's ``_parts_manager`` produces ``PartStartEvent`` /
``PartDeltaEvent``, and rbtr's event mapping translates both into
``TextDelta``.
"""

from __future__ import annotations

import queue
from collections.abc import AsyncIterator, Generator

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from rbtr.engine import Engine
from rbtr.events import Event, TextDelta, ToolCallFinished, ToolCallStarted
from rbtr.llm.agent import AgentDeps
from rbtr.llm.context import LLMContext
from rbtr.llm.stream import _do_stream
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

# ── Helpers ──────────────────────────────────────────────────────────


def _collect_text_deltas(events_q: queue.Queue[Event]) -> str:
    """Drain the queue and concatenate all ``TextDelta`` payloads."""
    parts: list[str] = []
    while True:
        try:
            event = events_q.get_nowait()
        except queue.Empty:
            break
        if isinstance(event, TextDelta):
            parts.append(event.delta)
    return "".join(parts)


def _final_text(messages: list[ModelMessage]) -> str:
    """Extract concatenated text from all ``TextPart``s in responses."""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    parts.append(part.content)
    return "".join(parts)


@pytest.fixture
def stream_ctx() -> Generator[tuple[LLMContext, AgentDeps, queue.Queue[Event]]]:
    """Lightweight context for ``_do_stream`` backed by a real engine."""
    state = EngineState()
    state.model_name = "test/test-model"
    events_q: queue.Queue[Event] = queue.Queue()
    with Engine(state, events_q, store=SessionStore()) as engine:
        # Sync store context so DB writes use the right session.
        engine._sync_store_context()
        ctx = engine._llm_context()
        deps = AgentDeps(state=state)
        yield ctx, deps, events_q


# ── Text fidelity tests ─────────────────────────────────────────────


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        # Single chunk — entire text arrives in PartStartEvent.
        (["Hello world!"], "Hello world!"),
        # Multiple chunks — first goes to PartStartEvent, rest to PartDeltaEvent.
        (["Ass", "umption: you want"], "Assumption: you want"),
        # Many small chunks — simulates token-by-token streaming.
        (["A", "B", "C", "D", "E"], "ABCDE"),
        # One large chunk.
        (["x" * 5000], "x" * 5000),
        # Unicode content.
        (["Hello ", "🌍", " — done"], "Hello 🌍 — done"),
    ],
    ids=[
        "single_chunk",
        "multi_chunk_first_token_lost_regression",
        "many_small_chunks",
        "one_large_chunk",
        "unicode",
    ],
)
async def test_streamed_text_matches_final_message(
    stream_ctx: tuple[LLMContext, AgentDeps, queue.Queue[Event]],
    chunks: list[str],
    expected: str,
) -> None:
    """Accumulated ``TextDelta`` events equal the finalized ``TextPart.content``."""
    ctx, deps, events_q = stream_ctx

    async def stream_fn(messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    model = FunctionModel(stream_function=stream_fn)
    result = await _do_stream(ctx, model, deps, None, "test", [])

    streamed = _collect_text_deltas(events_q)
    final = _final_text(result.all_messages)

    assert streamed == expected
    assert final == expected
    assert streamed == final


@pytest.mark.anyio
async def test_text_after_tool_calls_matches(
    stream_ctx: tuple[LLMContext, AgentDeps, queue.Queue[Event]],
) -> None:
    """Text emitted after a tool-call round-trip is fully captured.

    The stream function yields a tool call first, then text.
    The ``PartStartEvent`` for the post-tool-call text must emit
    a ``TextDelta`` so no characters are lost.
    """
    ctx, deps, events_q = stream_ctx

    call_count = 0

    async def stream_fn(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First request: tool call
            yield {0: DeltaToolCall(name="read_file", json_args='{"path": "a.py"}')}
        else:
            # Second request (after tool result): text response
            yield "Here"
            yield " are the"
            yield " results."

    model = FunctionModel(stream_function=stream_fn)
    result = await _do_stream(ctx, model, deps, None, "read a.py", [])

    # Collect all events
    all_events: list[Event] = []
    while True:
        try:
            all_events.append(events_q.get_nowait())
        except queue.Empty:
            break

    # There should be tool events AND text deltas
    tool_starts = [e for e in all_events if isinstance(e, ToolCallStarted)]
    tool_ends = [e for e in all_events if isinstance(e, ToolCallFinished)]
    text_deltas = [e for e in all_events if isinstance(e, TextDelta)]

    assert len(tool_starts) == 1
    assert len(tool_ends) == 1
    assert tool_starts[0].tool_name == "read_file"

    streamed = "".join(d.delta for d in text_deltas)
    final = _final_text(result.all_messages)

    assert streamed == "Here are the results."
    assert streamed == final


@pytest.mark.anyio
async def test_text_before_and_after_tool_call(
    stream_ctx: tuple[LLMContext, AgentDeps, queue.Queue[Event]],
) -> None:
    """Text before and after a tool call is fully captured.

    First model response has text + tool call.
    Second model response (after tool result) has only text.
    Both text segments must appear in ``TextDelta`` events.
    """
    ctx, deps, events_q = stream_ctx

    call_count = 0

    async def stream_fn(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Text preamble, then tool call
            yield "Let me check"
            yield {0: DeltaToolCall(name="read_file", json_args='{"path": "b.py"}')}
        else:
            yield "Found it: OK"

    model = FunctionModel(stream_function=stream_fn)
    result = await _do_stream(ctx, model, deps, None, "check b.py", [])

    all_events: list[Event] = []
    while True:
        try:
            all_events.append(events_q.get_nowait())
        except queue.Empty:
            break

    text_deltas = [e for e in all_events if isinstance(e, TextDelta)]
    streamed = "".join(d.delta for d in text_deltas)
    final = _final_text(result.all_messages)

    # Both preamble ("Let me check") and result ("Found it: OK") are captured
    assert "Let me check" in streamed
    assert "Found it: OK" in streamed
    assert streamed == final


@pytest.mark.anyio
async def test_db_content_matches_streamed_text(
    stream_ctx: tuple[LLMContext, AgentDeps, queue.Queue[Event]],
) -> None:
    """Text persisted to the DB matches what was streamed to the UI."""
    ctx, deps, events_q = stream_ctx

    async def stream_fn(messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[str]:
        yield "Database"
        yield " and UI"
        yield " must match."

    model = FunctionModel(stream_function=stream_fn)
    result = await _do_stream(ctx, model, deps, None, "test", [])

    streamed = _collect_text_deltas(events_q)
    final = _final_text(result.all_messages)

    # Load from DB
    db_messages = ctx.store.load_messages(ctx.state.session_id)
    db_text = _final_text(db_messages)

    assert streamed == "Database and UI must match."
    assert streamed == final
    assert streamed == db_text
