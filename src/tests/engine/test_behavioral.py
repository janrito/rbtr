"""End-to-end behavioral tests for session persistence.

Uses ``pydantic_ai.models.test.TestModel`` for a real agent loop —
no mocks. These tests define the contract that must survive the
part-level persistence refactor.

Tests verify observable outcomes: save/load round-trips, session
listing, history search, resume, compaction continuity.
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from rbtr.engine import Engine
from rbtr.events import Event, TaskFinished, ToolCallFinished, ToolCallStarted
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

from .conftest import drain, has_event_type, output_texts

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# ── Helpers ──────────────────────────────────────────────────────────


def _make_engine(*, store: SessionStore | None = None) -> Engine:
    """Build an engine wired to TestModel with an in-memory store."""
    state = EngineState(owner="testowner", repo_name="testrepo")
    state.openai_connected = True
    state.model_name = "openai/gpt-4o"
    eng = Engine(state, queue.Queue(), store=store or SessionStore())
    return eng


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


def test_multi_turn_roundtrip(mocker: MockerFixture, creds_path: Path) -> None:
    """Run 3 user turns, verify store.load_messages() returns the
    full conversation and messages round-trip correctly.
    """
    from pydantic_ai.models.test import TestModel

    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")

    engine = _make_engine()

    # TestModel with custom text output (no tool calls for simplicity).
    test_model = TestModel(custom_output_text="test response", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    try:
        prompts = ["first question", "second question", "third question"]
        for prompt in prompts:
            events = _run_llm_turn(engine, prompt)
            _assert_task_succeeded(events)

        # Verify all messages are in the store.
        loaded = engine.store.load_messages(engine.state.session_id)

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

    finally:
        engine.close()


def test_multi_turn_with_tool_calls(mocker: MockerFixture, creds_path: Path) -> None:
    """Run a turn with tool calls, verify tool call and return parts
    are persisted in the store.
    """
    from pydantic_ai.models.test import TestModel

    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")

    engine = _make_engine()

    # TestModel calls all tools on the first request, then responds with text.
    test_model = TestModel(custom_output_text="analysis complete", call_tools="all")
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    try:
        events = _run_llm_turn(engine, "analyse the code")
        _assert_task_succeeded(events)

        # Verify tool events were emitted.
        assert has_event_type(events, ToolCallStarted)
        assert has_event_type(events, ToolCallFinished)

        # Verify messages are in the store.
        loaded = engine.store.load_messages(engine.state.session_id)
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

    finally:
        engine.close()


# ── Session listing ──────────────────────────────────────────────────


def test_session_listing_after_turns(mocker: MockerFixture, creds_path: Path) -> None:
    """After a multi-turn run, list_sessions shows correct metadata."""
    from pydantic_ai.models.test import TestModel

    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")

    engine = _make_engine()
    test_model = TestModel(custom_output_text="reply", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    try:
        for prompt in ["q1", "q2"]:
            events = _run_llm_turn(engine, prompt)
            _assert_task_succeeded(events)

        sessions = engine.store.list_sessions()
        assert len(sessions) == 1

        session = sessions[0]
        assert session.session_id == engine.state.session_id
        assert session.message_count == 4  # 2 turns x 2 messages

    finally:
        engine.close()


# ── History search ───────────────────────────────────────────────────


def test_history_search_finds_user_prompts(mocker: MockerFixture, creds_path: Path) -> None:
    """search_history returns user prompts from the conversation."""
    from pydantic_ai.models.test import TestModel

    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")

    engine = _make_engine()
    test_model = TestModel(custom_output_text="ok", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    try:
        for prompt in ["review tui.py", "explain config"]:
            _run_llm_turn(engine, prompt)

        results = engine.store.search_history()
        assert "review tui.py" in results
        assert "explain config" in results
        # Assistant text should not appear in search results.
        assert "ok" not in results

    finally:
        engine.close()


# ── Session resume ───────────────────────────────────────────────────


def test_session_resume_loads_messages(mocker: MockerFixture, creds_path: Path) -> None:
    """Save a conversation, start a new session, resume the old one —
    verify loaded messages match.
    """
    from pydantic_ai.models.test import TestModel

    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")

    engine = _make_engine()
    test_model = TestModel(custom_output_text="hello back", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    try:
        # Turn 1 in session A.
        _run_llm_turn(engine, "hello")
        session_a_id = engine.state.session_id
        messages_a = engine.store.load_messages(session_a_id)

        # Start new session.
        engine.run_task("command", "/new")
        drain(engine.events)
        assert engine.state.session_id != session_a_id

        # Turn in session B.
        _run_llm_turn(engine, "different conversation")

        # Resume session A.
        engine.run_task("command", f"/session resume {session_a_id}")
        events = drain(engine.events)
        assert any("Resumed" in t for t in output_texts(events))
        assert engine.state.session_id == session_a_id
        assert len(engine.store.load_messages(session_a_id)) == len(messages_a)

    finally:
        engine.close()


# ── Compaction + resume ──────────────────────────────────────────────


def test_compaction_reduces_history(mocker: MockerFixture, creds_path: Path) -> None:
    """Run enough turns to compact, verify history is shorter after."""
    from pydantic_ai.models.test import TestModel

    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")

    engine = _make_engine()
    test_model = TestModel(custom_output_text="reply", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    try:
        # Build enough history to compact (15 turns).
        for i in range(15):
            _run_llm_turn(engine, f"question {i}")
            drain(engine.events)

        pre_compact_count = len(engine.store.load_messages(engine.state.session_id))
        assert pre_compact_count == 30  # 15 turns x 2

        # Mock the summary LLM call (compaction uses its own agent).
        mocker.patch(
            "rbtr.llm.compact._stream_summary",
            return_value="Summary: discussed 15 questions.",
        )
        mocker.patch("rbtr.llm.compact.build_model", return_value=test_model)

        engine.state.usage.context_window = 200_000
        engine.run_task("command", "/compact")
        drain(engine.events)

        post_compact = engine.store.load_messages(engine.state.session_id)
        assert len(post_compact) < pre_compact_count

        # Summary message is present.
        first = post_compact[0]
        assert isinstance(first, ModelRequest)
        assert any(
            isinstance(p, UserPromptPart) and isinstance(p.content, str) and "Summary" in p.content
            for p in first.parts
        )

    finally:
        engine.close()


def test_compaction_then_resume(mocker: MockerFixture, creds_path: Path) -> None:
    """After compaction, resume from a different session and verify
    the compacted state loads correctly.
    """
    from pydantic_ai.models.test import TestModel

    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")

    engine = _make_engine()
    test_model = TestModel(custom_output_text="reply", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    try:
        # Build history and compact.
        for i in range(15):
            _run_llm_turn(engine, f"q{i}")
            drain(engine.events)

        mocker.patch(
            "rbtr.llm.compact._stream_summary",
            return_value="Compacted summary.",
        )
        mocker.patch("rbtr.llm.compact.build_model", return_value=test_model)

        engine.state.usage.context_window = 200_000
        engine.run_task("command", "/compact")
        drain(engine.events)

        compacted_session_id = engine.state.session_id
        compacted_messages = engine.store.load_messages(compacted_session_id)

        # Switch to new session.
        engine.run_task("command", "/new")
        drain(engine.events)

        # Resume compacted session.
        engine.run_task("command", f"/session resume {compacted_session_id}")
        events = drain(engine.events)
        assert any("Resumed" in t for t in output_texts(events))

        # Loaded messages match the compacted state.
        assert len(engine.store.load_messages(compacted_session_id)) == len(compacted_messages)

    finally:
        engine.close()


# ── Compaction preserves continuity ──────────────────────────────────


def test_compaction_preserves_continuity(mocker: MockerFixture, creds_path: Path) -> None:
    """After compaction, sending another message works — the agent
    can respond using the compacted history.
    """
    from pydantic_ai.models.test import TestModel

    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")

    engine = _make_engine()
    test_model = TestModel(custom_output_text="reply", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    try:
        for i in range(15):
            _run_llm_turn(engine, f"q{i}")
            drain(engine.events)

        mocker.patch(
            "rbtr.llm.compact._stream_summary",
            return_value="Compacted summary of earlier conversation.",
        )
        mocker.patch("rbtr.llm.compact.build_model", return_value=test_model)

        engine.state.usage.context_window = 200_000
        engine.run_task("command", "/compact")
        drain(engine.events)

        # Send another message after compaction.
        events = _run_llm_turn(engine, "follow-up question")
        _assert_task_succeeded(events)

        # Verify the response was added.
        loaded = engine.store.load_messages(engine.state.session_id)
        user_texts = [
            p.content
            for msg in loaded
            if isinstance(msg, ModelRequest)
            for p in msg.parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        ]
        assert "follow-up question" in user_texts

    finally:
        engine.close()


# ── Command/shell rows don't pollute history ─────────────────────────


def test_command_shell_rows_excluded_from_history(mocker: MockerFixture, creds_path: Path) -> None:
    """Command and shell rows stored between LLM turns don't appear
    in load_messages().
    """
    from pydantic_ai.models.test import TestModel

    from rbtr.creds import creds

    creds.update(openai_api_key="sk-test")

    engine = _make_engine()
    test_model = TestModel(custom_output_text="ok", call_tools=[])
    mocker.patch("rbtr.llm.stream.build_model", return_value=test_model)

    try:
        # LLM turn.
        _run_llm_turn(engine, "hello")
        drain(engine.events)

        # Persist command/shell inputs via the store API — these
        # should be excluded from load_messages but visible in
        # search_history.
        engine.store.save_input(engine.state.session_id, "/help", "command")
        engine.store.save_input(engine.state.session_id, "!git status", "shell")

        # Another LLM turn.
        _run_llm_turn(engine, "world")
        drain(engine.events)

        # load_messages should only return LLM messages.
        loaded = engine.store.load_messages(engine.state.session_id)
        assert len(loaded) == 4  # 2 turns x 2 messages

        # But search_history should find the command.
        results = engine.store.search_history()
        assert "/help" in results
        assert "!git status" in results

    finally:
        engine.close()
