"""End-to-end behavioral tests for session persistence.

Uses ``pydantic_ai.models.test.TestModel`` for a real agent loop —
no mocks. These tests define the contract that must survive the
part-level persistence refactor.

Tests verify observable outcomes: save/load round-trips, session
listing, history search, resume, compaction continuity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from rbtr.engine import Engine
from rbtr.events import Event, TaskFinished, ToolCallFinished, ToolCallStarted

from .conftest import drain, has_event_type, output_texts, summary_result

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# ── Helpers ──────────────────────────────────────────────────────────


def _run_llm_turn(engine: Engine, message: str) -> list[Event]:
    """Send a user message through the LLM and return all events."""
    engine.run_task("llm", message)
    return drain(engine.events)


def _assert_task_succeeded(events: list[Event]) -> None:
    """Assert that a TaskFinished with success=True is present."""
    finished = [e for e in events if isinstance(e, TaskFinished)]
    assert finished, "No TaskFinished event found"
    assert finished[-1].success, f"Task failed: {finished[-1]}"


# ── Multi-turn with tools ───────────────────────────────────────────


def test_multi_turn_roundtrip(mocker: MockerFixture, llm_engine: Engine) -> None:
    """Run 3 user turns, verify store.load_messages() returns the
    full conversation and messages round-trip correctly.
    """
    test_model = TestModel(custom_output_text="test response", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    prompts = ["first question", "second question", "third question"]
    for prompt in prompts:
        events = _run_llm_turn(llm_engine, prompt)
        _assert_task_succeeded(events)

    # Verify all messages are in the store.
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)

    # Each turn = 1 request + 1 response = 2 messages, 3 turns = 6.
    assert len(loaded) == 6

    # Verify user prompts are preserved.
    user_texts = []
    for msg in loaded:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    user_texts.append(part.content)
    assert user_texts == prompts

    # Verify responses are preserved.
    response_texts = []
    for msg in loaded:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    response_texts.append(part.content)
    assert len(response_texts) == 3
    assert all("test response" in t for t in response_texts)


def test_multi_turn_with_tool_calls(mocker: MockerFixture, llm_engine: Engine) -> None:
    """Run a turn with tool calls, verify tool call and return parts
    are persisted in the store.
    """
    test_model = TestModel(custom_output_text="analysis complete", call_tools="all")
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    events = _run_llm_turn(llm_engine, "analyse the code")
    _assert_task_succeeded(events)

    # Verify tool events were emitted.
    assert has_event_type(events, ToolCallStarted)
    assert has_event_type(events, ToolCallFinished)

    # Verify messages are in the store.
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert (
        len(loaded) >= 3
    )  # request, response (tool calls), request (tool returns), response (text)

    # Verify tool call parts exist in the loaded messages.
    has_tool_call = any(
        isinstance(part, ToolCallPart)
        for msg in loaded
        if isinstance(msg, ModelResponse)
        for part in msg.parts
    )
    assert has_tool_call

    # Verify tool return parts exist.
    has_tool_return = any(
        isinstance(part, ToolReturnPart)
        for msg in loaded
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )
    assert has_tool_return


# ── Session listing ──────────────────────────────────────────────────


def test_session_listing_after_turns(mocker: MockerFixture, llm_engine: Engine) -> None:
    """After a multi-turn run, list_sessions shows correct metadata."""
    test_model = TestModel(custom_output_text="reply", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    for prompt in ["q1", "q2"]:
        events = _run_llm_turn(llm_engine, prompt)
        _assert_task_succeeded(events)

    sessions = llm_engine.store.list_sessions()
    assert len(sessions) == 1

    session = sessions[0]
    assert session.session_id == llm_engine.state.session_id
    assert session.message_count == 4  # 2 turns x 2 messages


# ── History search ───────────────────────────────────────────────────


def test_history_search_finds_user_prompts(mocker: MockerFixture, llm_engine: Engine) -> None:
    """search_history returns user prompts from the conversation."""
    test_model = TestModel(custom_output_text="ok", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    for prompt in ["review tui.py", "explain config"]:
        _run_llm_turn(llm_engine, prompt)

    results = llm_engine.store.search_history()
    assert "review tui.py" in results
    assert "explain config" in results
    # Assistant text should not appear in search results.
    assert "ok" not in results


# ── Session resume ───────────────────────────────────────────────────


def test_session_resume_loads_messages(mocker: MockerFixture, llm_engine: Engine) -> None:
    """Save a conversation, start a new session, resume the old one —
    verify loaded messages match.
    """
    test_model = TestModel(custom_output_text="hello back", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    # Turn 1 in session A.
    _run_llm_turn(llm_engine, "hello")
    session_a_id = llm_engine.state.session_id
    messages_a = llm_engine.store.load_messages(session_a_id)

    # Start new session.
    llm_engine.run_task("command", "/new")
    drain(llm_engine.events)
    assert llm_engine.state.session_id != session_a_id

    # Turn in session B.
    _run_llm_turn(llm_engine, "different conversation")

    # Resume session A.
    llm_engine.run_task("command", f"/session resume {session_a_id}")
    events = drain(llm_engine.events)
    assert any("Resumed" in t for t in output_texts(events))
    assert llm_engine.state.session_id == session_a_id
    assert len(llm_engine.store.load_messages(session_a_id)) == len(messages_a)


# ── Compaction + resume ──────────────────────────────────────────────


def test_compaction_reduces_history(mocker: MockerFixture, llm_engine: Engine) -> None:
    """Run enough turns to compact, verify history is shorter after."""
    test_model = TestModel(custom_output_text="reply", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    # Build enough history to compact (15 turns).
    for i in range(15):
        _run_llm_turn(llm_engine, f"question {i}")
        drain(llm_engine.events)

    pre_compact_count = len(llm_engine.store.load_messages(llm_engine.state.session_id))
    assert pre_compact_count == 30  # 15 turns x 2

    # Mock the summary LLM call (compaction uses its own agent).
    mocker.patch(
        "rbtr.llm.compact._stream_summary",
        return_value=summary_result("Summary: discussed 15 questions."),
    )
    mocker.patch("rbtr.llm.compact.build_model", return_value=test_model)
    mocker.patch("rbtr.llm.compact.run_fact_extraction", return_value=None)

    llm_engine.state.usage.context_window = 200_000
    llm_engine.run_task("command", "/compact")
    drain(llm_engine.events)

    post_compact = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert len(post_compact) < pre_compact_count

    # Summary message is present.
    first = post_compact[0]
    assert isinstance(first, ModelRequest)
    assert any(
        isinstance(p, UserPromptPart) and isinstance(p.content, str) and "Summary" in p.content
        for p in first.parts
    )


def test_compaction_then_resume(mocker: MockerFixture, llm_engine: Engine) -> None:
    """After compaction, resume from a different session and verify
    the compacted state loads correctly.
    """
    test_model = TestModel(custom_output_text="reply", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    # Build history and compact.
    for i in range(15):
        _run_llm_turn(llm_engine, f"q{i}")
        drain(llm_engine.events)

    mocker.patch(
        "rbtr.llm.compact._stream_summary",
        return_value=summary_result("Compacted summary."),
    )
    mocker.patch("rbtr.llm.compact.build_model", return_value=test_model)

    llm_engine.state.usage.context_window = 200_000
    llm_engine.run_task("command", "/compact")
    drain(llm_engine.events)

    compacted_session_id = llm_engine.state.session_id
    compacted_messages = llm_engine.store.load_messages(compacted_session_id)

    # Switch to new session.
    llm_engine.run_task("command", "/new")
    drain(llm_engine.events)

    # Resume compacted session.
    llm_engine.run_task("command", f"/session resume {compacted_session_id}")
    events = drain(llm_engine.events)
    assert any("Resumed" in t for t in output_texts(events))

    # Loaded messages match the compacted state.
    assert len(llm_engine.store.load_messages(compacted_session_id)) == len(compacted_messages)


# ── Compaction preserves continuity ──────────────────────────────────


def test_compaction_preserves_continuity(mocker: MockerFixture, llm_engine: Engine) -> None:
    """After compaction, sending another message works — the agent
    can respond using the compacted history.
    """
    test_model = TestModel(custom_output_text="reply", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    for i in range(15):
        _run_llm_turn(llm_engine, f"q{i}")
        drain(llm_engine.events)

    mocker.patch(
        "rbtr.llm.compact._stream_summary",
        return_value=summary_result("Compacted summary of earlier conversation."),
    )
    mocker.patch("rbtr.llm.compact.build_model", return_value=test_model)

    llm_engine.state.usage.context_window = 200_000
    llm_engine.run_task("command", "/compact")
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


# ── Command/shell rows don't pollute history ─────────────────────────


def test_command_shell_rows_excluded_from_history(
    mocker: MockerFixture, llm_engine: Engine
) -> None:
    """Command and shell rows stored between LLM turns don't appear
    in load_messages().
    """
    test_model = TestModel(custom_output_text="ok", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    # LLM turn.
    _run_llm_turn(llm_engine, "hello")
    drain(llm_engine.events)

    # Persist command/shell inputs via the store API — these
    # should be excluded from load_messages but visible in
    # search_history.
    llm_engine.store.save_input(llm_engine.state.session_id, "/help", "command")
    llm_engine.store.save_input(llm_engine.state.session_id, "!git status", "shell")

    # Another LLM turn.
    _run_llm_turn(llm_engine, "world")
    drain(llm_engine.events)

    # load_messages should only return LLM messages.
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert len(loaded) == 4  # 2 turns x 2 messages

    # But search_history should find the command.
    results = llm_engine.store.search_history()
    assert "/help" in results
    assert "!git status" in results
