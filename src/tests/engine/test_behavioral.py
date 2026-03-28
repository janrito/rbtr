"""End-to-end behavioral tests for session persistence.

Uses `pydantic_ai.models.test.TestModel` for a real agent loop —
no mocks. These tests define the contract that must survive the
part-level persistence refactor.

Tests verify observable outcomes: save/load round-trips, session
listing, history search, resume, compaction continuity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from rbtr.engine.core import Engine
from rbtr.engine.types import TaskType
from rbtr.events import Event, TaskFinished
from rbtr.llm.compact import compact_agent
from tests.helpers import StubProvider, drain, output_texts
from tests.sessions.assertions import assert_ordering

if TYPE_CHECKING:
    pass


# ── Helpers ──────────────────────────────────────────────────────────


def _run_llm_turn(engine: Engine, message: str) -> list[Event]:
    """Send a user message through the LLM and return all events."""
    engine.run_task(TaskType.LLM, message)
    return drain(engine.events)


def _assert_task_succeeded(events: list[Event]) -> None:
    """Assert that a TaskFinished with success=True is present."""
    finished = [e for e in events if isinstance(e, TaskFinished)]
    assert finished, "No TaskFinished event found"
    assert finished[-1].success, f"Task failed: {finished[-1]}"


# ── Session listing ──────────────────────────────────────────────────


def test_session_listing_after_turns(llm_engine: Engine, stub_provider: StubProvider) -> None:
    """After a multi-turn run, list_sessions shows correct metadata."""
    stub_provider.set_model(TestModel(custom_output_text="reply", call_tools=[]))

    for prompt in ["q1", "q2"]:
        events = _run_llm_turn(llm_engine, prompt)
        _assert_task_succeeded(events)

    sessions = llm_engine.store.list_sessions()
    assert len(sessions) == 1

    session = sessions[0]
    assert session.session_id == llm_engine.state.session_id
    assert session.message_count == 4  # 2 turns x 2 messages


# ── History search ───────────────────────────────────────────────────


def test_history_search_finds_user_prompts(llm_engine: Engine, stub_provider: StubProvider) -> None:
    """search_history returns user prompts from the conversation."""
    stub_provider.set_model(TestModel(custom_output_text="ok", call_tools=[]))

    for prompt in ["review tui.py", "explain config"]:
        _run_llm_turn(llm_engine, prompt)

    results = llm_engine.store.search_history()
    assert "review tui.py" in results
    assert "explain config" in results
    # Assistant text should not appear in search results.
    assert "ok" not in results


# ── Session resume ───────────────────────────────────────────────────


def test_session_resume_loads_messages(llm_engine: Engine, stub_provider: StubProvider) -> None:
    """Save a conversation, start a new session, resume the old one —
    verify loaded messages match.
    """
    stub_provider.set_model(TestModel(custom_output_text="hello back", call_tools=[]))

    # Turn 1 in session A.
    _run_llm_turn(llm_engine, "hello")
    session_a_id = llm_engine.state.session_id
    messages_a = llm_engine.store.load_messages(session_a_id)

    # Start new session.
    llm_engine.run_task(TaskType.COMMAND, "/new")
    drain(llm_engine.events)
    assert llm_engine.state.session_id != session_a_id

    # Turn in session B.
    _run_llm_turn(llm_engine, "different conversation")

    # Resume session A.
    llm_engine.run_task(TaskType.COMMAND, f"/session resume {session_a_id}")
    events = drain(llm_engine.events)
    assert any("Resumed" in t for t in output_texts(events))
    assert llm_engine.state.session_id == session_a_id
    assert len(llm_engine.store.load_messages(session_a_id)) == len(messages_a)

    # Send another turn after resume — conversation continues.
    events = _run_llm_turn(llm_engine, "follow-up after resume")
    _assert_task_succeeded(events)

    loaded = llm_engine.store.load_messages(session_a_id)
    assert_ordering(loaded)
    user_texts = [
        p.content
        for m in loaded
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart) and isinstance(p.content, str)
    ]
    assert "hello" in user_texts
    assert "follow-up after resume" in user_texts


# ── Compaction + resume ──────────────────────────────────────────────


def test_compaction_then_resume(llm_engine: Engine, stub_provider: StubProvider) -> None:
    """After compaction, resume from a different session and verify
    the compacted state loads correctly.
    """
    stub_provider.set_model(TestModel(custom_output_text="reply", call_tools=[]))

    # Build history and compact.
    for i in range(15):
        _run_llm_turn(llm_engine, f"q{i}")
        drain(llm_engine.events)

    llm_engine.state.usage.context_window = 200_000
    with compact_agent.override(model=TestModel(custom_output_text="Compacted summary.")):
        llm_engine.run_task(TaskType.COMMAND, "/compact")
    drain(llm_engine.events)

    compacted_session_id = llm_engine.state.session_id
    compacted_messages = llm_engine.store.load_messages(compacted_session_id)

    # Switch to new session.
    llm_engine.run_task(TaskType.COMMAND, "/new")
    drain(llm_engine.events)

    # Resume compacted session.
    llm_engine.run_task(TaskType.COMMAND, f"/session resume {compacted_session_id}")
    events = drain(llm_engine.events)
    assert any("Resumed" in t for t in output_texts(events))

    # Loaded messages match the compacted state.
    resumed = llm_engine.store.load_messages(compacted_session_id)
    assert len(resumed) == len(compacted_messages)
    assert_ordering(resumed)

    # Summary text survived the resume.
    assert any(
        isinstance(p, UserPromptPart)
        and isinstance(p.content, str)
        and "Compacted summary" in p.content
        for msg in resumed
        if isinstance(msg, ModelRequest)
        for p in msg.parts
    )


# ── Compaction preserves continuity ──────────────────────────────────


def test_compaction_preserves_continuity(llm_engine: Engine, stub_provider: StubProvider) -> None:
    """After compaction, sending another message works — the agent
    can respond using the compacted history.
    """
    stub_provider.set_model(TestModel(custom_output_text="reply", call_tools=[]))

    for i in range(15):
        _run_llm_turn(llm_engine, f"q{i}")
        drain(llm_engine.events)

    llm_engine.state.usage.context_window = 200_000
    with compact_agent.override(
        model=TestModel(custom_output_text="Compacted summary of earlier conversation.")
    ):
        llm_engine.run_task(TaskType.COMMAND, "/compact")
    drain(llm_engine.events)

    # Send another message after compaction.
    events = _run_llm_turn(llm_engine, "follow-up question")
    _assert_task_succeeded(events)

    # Verify the response was added.
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    user_texts = [
        p.content
        for msg in loaded
        if isinstance(msg, ModelRequest)
        for p in msg.parts
        if isinstance(p, UserPromptPart) and isinstance(p.content, str)
    ]
    assert "follow-up question" in user_texts
    assert_ordering(loaded)

    # Summary from compaction is still present.
    assert any("Compacted summary" in t for t in user_texts)
