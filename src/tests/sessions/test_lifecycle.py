"""End-to-end lifecycle tests: stream → persist → reload → verify.

These tests run the real `_do_stream` pipeline with a
`FunctionModel` and an in-memory `SessionStore`.  After the
stream completes, messages are loaded from the DB and compared
against the original model output.

The key invariants:

1. **Ordering**: every `ModelRequest` precedes its
   `ModelResponse` in the loaded history.
2. **Fidelity**: loaded messages match the originals field by
   field — including `ToolCallPart.tool_call_id` and
   `ToolReturnPart.content`.
3. **Pairing**: every `ToolCallPart` has a matching
   `ToolReturnPart` (same `tool_call_id`) in the immediately
   following `ModelRequest`.
4. **Alternation**: no two consecutive `ModelResponse`s (the
   provider would reject).

Each test scenario models a real conversation shape that has
caused production bugs.
"""

from __future__ import annotations

import contextlib
import queue
from collections.abc import AsyncIterator, Generator

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from rbtr.engine.core import Engine
from rbtr.events import Event
from rbtr.llm.context import LLMContext
from rbtr.llm.deps import AgentDeps
from rbtr.llm.stream import _do_stream
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

from .conftest import assert_messages_match, assert_ordering, assert_tool_pairing

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def llm_ctx() -> Generator[LLMContext]:
    """LLMContext backed by an in-memory store."""
    state = EngineState()
    state.model_name = "test/lifecycle"
    events_q: queue.Queue[Event] = queue.Queue()
    with Engine(state, events_q, store=SessionStore()) as engine:
        engine._sync_store_context()
        yield engine._llm_context()


# ── Happy path ───────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_text_response(llm_ctx: LLMContext) -> None:
    """User prompt → text response: request before response on reload."""
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "Hello! Here's my review."

    model = FunctionModel(stream_function=stream_fn)
    await _do_stream(llm_ctx, model, deps, None, "review my code", [])
    loaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)

    assert len(loaded) == 2
    assert isinstance(loaded[0], ModelRequest)
    assert isinstance(loaded[1], ModelResponse)
    assert loaded[0].parts[0].content == "review my code"  # type: ignore[union-attr]
    assert loaded[1].parts[0].content == "Hello! Here's my review."  # type: ignore[union-attr]
    assert_ordering(loaded)


@pytest.mark.anyio
async def test_tool_call_and_return(llm_ctx: LLMContext) -> None:
    """Tool call → tool return → text: pairing correct on reload."""
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)
    call_count = 0

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield "Let me check."
            yield {0: DeltaToolCall(name="read_file", json_args='{"path": "a.py"}')}
        else:
            yield "Here's the file content."

    model = FunctionModel(stream_function=stream_fn)
    await _do_stream(llm_ctx, model, deps, None, "read a.py", [])
    loaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)

    assert len(loaded) == 4
    assert_ordering(loaded)
    assert_tool_pairing(loaded)


@pytest.mark.anyio
async def test_multiple_parallel_tool_calls(llm_ctx: LLMContext) -> None:
    """Response with 2 tool calls — both returns paired on reload."""
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)
    call_count = 0

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {
                0: DeltaToolCall(name="read_file", json_args='{"path": "a.py"}'),
                1: DeltaToolCall(name="read_file", json_args='{"path": "b.py"}'),
            }
        else:
            yield "Both files read."

    model = FunctionModel(stream_function=stream_fn)
    await _do_stream(llm_ctx, model, deps, None, "read both files", [])
    loaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)

    assert_ordering(loaded)
    assert_tool_pairing(loaded)

    resp = loaded[1]
    assert isinstance(resp, ModelResponse)
    call_ids = {p.tool_call_id for p in resp.parts if isinstance(p, ToolCallPart)}
    assert len(call_ids) == 2


@pytest.mark.anyio
async def test_multi_turn(llm_ctx: LLMContext) -> None:
    """Two user turns — ordering preserved on reload."""
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "Response."

    model = FunctionModel(stream_function=stream_fn)

    result1 = await _do_stream(llm_ctx, model, deps, None, "question 1", [])
    await _do_stream(llm_ctx, model, deps, None, "question 2", result1.all_messages)

    loaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert len(loaded) == 4
    assert_ordering(loaded)
    assert loaded[0].parts[0].content == "question 1"  # type: ignore[union-attr]
    assert loaded[2].parts[0].content == "question 2"  # type: ignore[union-attr]


# ── Session resume ───────────────────────────────────────────────────


@pytest.mark.anyio
async def test_resume_ordering(llm_ctx: LLMContext) -> None:
    """Persist turn 1, reload from DB, persist turn 2.

    Request before response at every position.
    """
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "Response."

    model = FunctionModel(stream_function=stream_fn)

    await _do_stream(llm_ctx, model, deps, None, "turn 1", [])

    reloaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert_ordering(reloaded)

    await _do_stream(llm_ctx, model, deps, None, "turn 2", reloaded)

    final = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert len(final) == 4
    assert_ordering(final)
    assert final[0].parts[0].content == "turn 1"  # type: ignore[union-attr]
    assert final[2].parts[0].content == "turn 2"  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_resume_after_tool_call_turn(llm_ctx: LLMContext) -> None:
    """Resume after a tool-calling turn — pairing survives reload."""
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)
    call_count = 0

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {0: DeltaToolCall(name="read_file", json_args='{"path": "a.py"}')}
        elif call_count == 2:
            yield "File read."
        else:
            yield "Follow-up."

    model = FunctionModel(stream_function=stream_fn)

    await _do_stream(llm_ctx, model, deps, None, "read a.py", [])

    reloaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert_ordering(reloaded)
    assert_tool_pairing(reloaded)

    await _do_stream(llm_ctx, model, deps, None, "now explain it", reloaded)

    final = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert_ordering(final)
    assert_tool_pairing(final)
    assert len(final) == 6


@pytest.mark.anyio
async def test_resumed_history_reaches_model_intact(llm_ctx: LLMContext) -> None:
    """The history the model receives on resume is exactly what was
    persisted — same messages, same order, same part types and content.

    Turn 1 streams normally.  Turn 2 captures the `messages`
    argument the model receives and compares it against what
    `load_messages` returned.
    """
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)
    received_history: list[list[ModelMessage]] = []
    call_count = 0

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        received_history.append(list(messages))
        yield f"Response {call_count}."

    model = FunctionModel(stream_function=stream_fn)

    await _do_stream(llm_ctx, model, deps, None, "first question", [])

    reloaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert_ordering(reloaded)

    await _do_stream(llm_ctx, model, deps, None, "second question", reloaded)

    assert len(received_history) == 2
    turn2_messages = received_history[1]

    # The reloaded history is the prefix of what the model saw.
    assert_messages_match(reloaded, turn2_messages)
    assert isinstance(turn2_messages[0], ModelRequest)
    assert_ordering(turn2_messages)


# ── Failure paths ────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_model_error_mid_stream(llm_ctx: LLMContext) -> None:
    """Model raises after partial output — request visible, response excluded."""
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "Partial content…"
        raise RuntimeError("model connection lost")

    model = FunctionModel(stream_function=stream_fn)

    with pytest.raises(RuntimeError, match="model connection lost"):
        await _do_stream(llm_ctx, model, deps, None, "hello", [])

    loaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert len(loaded) == 1
    assert isinstance(loaded[0], ModelRequest)
    assert loaded[0].parts[0].content == "hello"  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_error_after_successful_turn(llm_ctx: LLMContext) -> None:
    """Turn 1 succeeds, turn 2 crashes mid-stream — turn 1 intact."""
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)

    async def good_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "All good."

    async def bad_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "About to crash…"
        raise RuntimeError("boom")

    good_model = FunctionModel(stream_function=good_fn)
    bad_model = FunctionModel(stream_function=bad_fn)

    result1 = await _do_stream(llm_ctx, good_model, deps, None, "turn 1", [])

    with pytest.raises(RuntimeError, match="boom"):
        await _do_stream(llm_ctx, bad_model, deps, None, "turn 2", result1.all_messages)

    loaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert len(loaded) == 3
    assert_ordering(loaded)
    assert loaded[0].parts[0].content == "turn 1"  # type: ignore[union-attr]
    assert isinstance(loaded[2], ModelRequest)
    assert loaded[2].parts[0].content == "turn 2"  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_error_during_tool_execution(llm_ctx: LLMContext) -> None:
    """Model crashes after tool call round — loaded history is valid."""
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)
    call_count = 0

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {0: DeltaToolCall(name="read_file", json_args='{"path": "a.py"}')}
        else:
            raise RuntimeError("model died after tool execution")
            yield ""  # unreachable — makes this an async generator

    model = FunctionModel(stream_function=stream_fn)

    with pytest.raises(RuntimeError, match="model died"):
        await _do_stream(llm_ctx, model, deps, None, "read a.py", [])

    loaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    if loaded:
        assert_ordering(loaded)


@pytest.mark.anyio
async def test_model_error_before_first_yield(llm_ctx: LLMContext) -> None:
    """Model raises before yielding — nothing visible."""
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        raise RuntimeError("immediate failure")
        yield ""  # unreachable — makes this an async generator

    model = FunctionModel(stream_function=stream_fn)

    with pytest.raises(RuntimeError, match="immediate failure"):
        await _do_stream(llm_ctx, model, deps, None, "hello", [])

    loaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert loaded == []


@pytest.mark.anyio
async def test_resume_after_crashed_turn(llm_ctx: LLMContext) -> None:
    """Turn 1 OK → turn 2 crashes before yield → turn 3 OK.

    Turn 2 excluded entirely. Only turns 1 and 3 visible.
    """
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)

    async def good_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        yield "Response."

    async def bad_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        raise RuntimeError("crash")
        yield ""  # unreachable — makes this an async generator

    good_model = FunctionModel(stream_function=good_fn)
    bad_model = FunctionModel(stream_function=bad_fn)

    await _do_stream(llm_ctx, good_model, deps, None, "turn 1", [])

    reloaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    with pytest.raises(RuntimeError):
        await _do_stream(llm_ctx, bad_model, deps, None, "turn 2", reloaded)

    reloaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert_ordering(reloaded)
    assert len(reloaded) == 2

    await _do_stream(llm_ctx, good_model, deps, None, "turn 3", reloaded)

    final = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    assert_ordering(final)
    assert len(final) == 4
    texts = [
        p.content  # type: ignore[union-attr]
        for m in final
        if isinstance(m, ModelRequest)
        for p in m.parts
        if hasattr(p, "content")
    ]
    assert "turn 1" in texts
    assert "turn 3" in texts
    assert "turn 2" not in texts


@pytest.mark.anyio
async def test_empty_response(llm_ctx: LLMContext) -> None:
    """Model returns empty text — no corruption regardless of outcome."""
    deps = AgentDeps(state=llm_ctx.state, store=llm_ctx.store)
    call_count = 0

    async def stream_fn(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            yield ""
        else:
            yield "Finally a real response."

    model = FunctionModel(stream_function=stream_fn)

    with contextlib.suppress(Exception):
        await _do_stream(llm_ctx, model, deps, None, "hello", [])

    loaded = llm_ctx.store.load_messages(llm_ctx.state.session_id)
    if loaded:
        assert_ordering(loaded)
        for i, msg in enumerate(loaded):
            assert len(msg.parts) > 0, f"message {i} has empty parts"
