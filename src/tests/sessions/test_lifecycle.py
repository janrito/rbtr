"""End-to-end lifecycle tests: engine.run_task → events + DB.

Tests go through `engine.run_task` with the test provider.
After each task, assertions verify that the DB contains exactly
the messages exchanged — prompts sent and responses received —
in the correct order.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from rbtr.engine.core import Engine
from rbtr.engine.types import TaskType
from rbtr.events import TaskFinished, TextDelta
from tests.engine.test_compact import ALL_HISTORIES
from tests.helpers import TestProvider, drain

from .assertions import assert_ordering, assert_tool_pairing


def _user_texts(messages: list[ModelMessage]) -> list[str]:
    """Extract user prompt strings from loaded messages."""
    texts: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for p in msg.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    texts.append(p.content)
    return texts


def _response_texts(messages: list[ModelMessage]) -> list[str]:
    """Extract response text strings from loaded messages."""
    texts: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for p in msg.parts:
                if isinstance(p, TextPart):
                    texts.append(p.content)
    return texts


# ── Text responses ───────────────────────────────────────────────────


def test_prompt_and_response_persisted(llm_engine: Engine, test_provider: TestProvider) -> None:
    """User prompt and model response are both persisted to DB."""
    test_provider.set_model(TestModel(custom_output_text="Here's my review."))

    llm_engine.run_task(TaskType.LLM, "review my code")
    events = drain(llm_engine.events)
    assert isinstance(events[-1], TaskFinished)
    assert events[-1].success

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert_ordering(loaded)
    assert "review my code" in _user_texts(loaded)
    assert "Here's my review." in _response_texts(loaded)


def test_streamed_text_matches_db(llm_engine: Engine, test_provider: TestProvider) -> None:
    """TextDelta events streamed to UI match the text stored in DB."""

    async def _stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield "Hello "
        yield "world!"

    test_provider.set_model(FunctionModel(stream_function=_stream))

    llm_engine.run_task(TaskType.LLM, "greet me")
    events = drain(llm_engine.events)

    streamed = "".join(e.delta for e in events if isinstance(e, TextDelta))
    db_texts = _response_texts(llm_engine.store.load_messages(llm_engine.state.session_id))

    assert streamed == "Hello world!"
    assert "Hello world!" in db_texts


# ── Multi-turn ───────────────────────────────────────────────────────


def test_multi_turn_preserves_order(llm_engine: Engine, test_provider: TestProvider) -> None:
    """Two turns persist in order: prompt1, response1, prompt2, response2."""
    test_provider.set_model(TestModel(custom_output_text="Noted."))

    llm_engine.run_task(TaskType.LLM, "first question")
    drain(llm_engine.events)
    llm_engine.run_task(TaskType.LLM, "second question")
    drain(llm_engine.events)

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert_ordering(loaded)

    prompts = _user_texts(loaded)
    assert prompts[0] == "first question"
    assert prompts[1] == "second question"


def test_resume_continues_conversation(llm_engine: Engine, test_provider: TestProvider) -> None:
    """Turn 2 sees turn 1 history — model receives prior messages."""
    received: list[list[ModelMessage]] = []

    async def _capture(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        received.append(list(messages))
        yield "Response."

    test_provider.set_model(FunctionModel(stream_function=_capture))

    llm_engine.run_task(TaskType.LLM, "turn 1")
    drain(llm_engine.events)
    after_turn1 = llm_engine.store.load_messages(llm_engine.state.session_id)

    llm_engine.run_task(TaskType.LLM, "turn 2")
    drain(llm_engine.events)

    # Turn 2's model call received turn 1 as history prefix.
    assert len(received) == 2
    turn2_history = received[1]
    assert len(turn2_history) >= len(after_turn1)
    assert_ordering(turn2_history)

    # Model saw the actual turn 1 prompt and response content.
    turn2_texts = [
        p.content
        for m in turn2_history
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart) and isinstance(p.content, str)
    ]
    assert "turn 1" in turn2_texts


# ── Tool calls ───────────────────────────────────────────────────────


def test_tool_calls_paired_in_db(llm_engine: Engine, test_provider: TestProvider) -> None:
    """Tool call and tool return are correctly paired after round-trip."""
    test_provider.set_model(TestModel(call_tools="all"))

    llm_engine.run_task(TaskType.LLM, "check the code")
    drain(llm_engine.events)

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert_ordering(loaded)
    assert_tool_pairing(loaded)


def test_tool_call_then_text_resume(llm_engine: Engine, test_provider: TestProvider) -> None:
    """After a tool-calling turn, a text-only follow-up works."""
    test_provider.set_model(TestModel(call_tools="all"))
    llm_engine.run_task(TaskType.LLM, "read the file")
    drain(llm_engine.events)
    assert_tool_pairing(llm_engine.store.load_messages(llm_engine.state.session_id))

    test_provider.set_model(TestModel(custom_output_text="Follow-up."))
    llm_engine.run_task(TaskType.LLM, "now explain it")
    drain(llm_engine.events)

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert_ordering(loaded)
    assert_tool_pairing(loaded)
    assert "Follow-up." in _response_texts(loaded)


# ── Error recovery ───────────────────────────────────────────────────


def test_model_error_fails_task(llm_engine: Engine, test_provider: TestProvider) -> None:
    """Model error → task fails, no partial response in DB."""

    async def _fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("connection lost")
        yield ""

    test_provider.set_model(FunctionModel(stream_function=_fail))

    llm_engine.run_task(TaskType.LLM, "hello")
    events = drain(llm_engine.events)

    finished = [e for e in events if isinstance(e, TaskFinished)]
    assert finished
    assert not finished[-1].success


def test_error_does_not_corrupt_prior_turn(llm_engine: Engine, test_provider: TestProvider) -> None:
    """Turn 1 succeeds, turn 2 crashes — turn 1 intact in DB."""
    test_provider.set_model(TestModel(custom_output_text="All good."))
    llm_engine.run_task(TaskType.LLM, "turn 1")
    drain(llm_engine.events)

    async def _fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("boom")
        yield ""

    test_provider.set_model(FunctionModel(stream_function=_fail))
    llm_engine.run_task(TaskType.LLM, "turn 2")
    drain(llm_engine.events)

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert_ordering(loaded)
    assert "turn 1" in _user_texts(loaded)
    assert "All good." in _response_texts(loaded)


def test_resume_after_crash(llm_engine: Engine, test_provider: TestProvider) -> None:
    """Turn 1 OK → turn 2 crashes → turn 3 OK. Conversation recovers."""
    test_provider.set_model(TestModel(custom_output_text="Response."))
    llm_engine.run_task(TaskType.LLM, "turn 1")
    drain(llm_engine.events)

    async def _fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("crash")
        yield ""

    test_provider.set_model(FunctionModel(stream_function=_fail))
    llm_engine.run_task(TaskType.LLM, "turn 2")
    drain(llm_engine.events)

    test_provider.set_model(TestModel(custom_output_text="Recovered."))
    llm_engine.run_task(TaskType.LLM, "turn 3")
    drain(llm_engine.events)

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert_ordering(loaded)
    prompts = _user_texts(loaded)
    assert "turn 1" in prompts
    assert "turn 3" in prompts
    assert "Recovered." in _response_texts(loaded)


# ── All history shapes as prior context ──────────────────────────────


@pytest.mark.parametrize("name", list(ALL_HISTORIES.keys()))
def test_engine_handles_all_history_shapes(
    name: str, llm_engine: Engine, test_provider: TestProvider
) -> None:
    """Seed each history shape, send a new turn — pipeline handles it."""
    history = ALL_HISTORIES[name]
    llm_engine.store.save_messages(llm_engine.state.session_id, list(history))

    llm_engine.run_task(TaskType.LLM, "follow-up question")
    events = drain(llm_engine.events)

    finished = [e for e in events if isinstance(e, TaskFinished)]
    assert finished, f"{name}: no TaskFinished"
    assert finished[-1].success, f"{name}: task failed"

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert_ordering(loaded)
    assert "follow-up question" in _user_texts(loaded)
