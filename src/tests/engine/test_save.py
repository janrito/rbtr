"""Tests for session auto-save (engine → store integration)."""

from __future__ import annotations

import queue

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from rbtr.engine import Engine, Session
from rbtr.engine.save import save_new_messages
from rbtr.events import Event
from rbtr.sessions.store import SessionStore


def _make_engine() -> tuple[Engine, SessionStore]:
    """Engine with an in-memory store, pre-populated session."""
    session = Session(owner="acme", repo_name="app", model_name="claude/sonnet")
    events: queue.Queue[Event] = queue.Queue()
    store = SessionStore()  # in-memory
    engine = Engine(session, events, store=store)
    # Set label like _run_setup would.
    session.session_label = "acme/app — main"
    return engine, store


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=text)],
        usage=RequestUsage(input_tokens=100, output_tokens=50),
        model_name="test",
    )


# ── save_new_messages ────────────────────────────────────────────────


def test_save_persists_new_messages() -> None:
    """After save_new_messages, messages appear in the store."""
    engine, store = _make_engine()
    engine.session.message_history = [_user("hello"), _assistant("hi")]

    save_new_messages(engine, run_cost=0.01)

    loaded = store.load_messages(engine.session.session_id)
    assert len(loaded) == 2
    # Compare content, not full equality (tz repr differs after roundtrip).
    assert isinstance(loaded[0], ModelRequest)
    assert loaded[0].parts[0].content == "hello"  # type: ignore[union-attr]  # UserPromptPart


def test_save_is_incremental() -> None:
    """Second save only inserts new messages, not duplicates."""
    engine, store = _make_engine()
    engine.session.message_history = [_user("q1"), _assistant("a1")]
    save_new_messages(engine)

    engine.session.message_history.extend([_user("q2"), _assistant("a2")])
    save_new_messages(engine)

    loaded = store.load_messages(engine.session.session_id)
    assert len(loaded) == 4


def test_save_noop_when_nothing_new() -> None:
    """Calling save when nothing is new doesn't error or duplicate."""
    engine, store = _make_engine()
    engine.session.message_history = [_user("q1")]
    save_new_messages(engine)
    save_new_messages(engine)  # should be no-op

    loaded = store.load_messages(engine.session.session_id)
    assert len(loaded) == 1


def test_save_attributes_cost_to_last_message() -> None:
    """run_cost is written to the store on the last row."""
    engine, store = _make_engine()
    engine.session.message_history = [_user("q"), _assistant("a")]
    save_new_messages(engine, run_cost=0.05)

    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].total_cost == 0.05


# ── /new resets saved count ──────────────────────────────────────────


def test_new_resets_session_id() -> None:
    """After /new, a new session_id is assigned."""
    engine, store = _make_engine()
    old_id = engine.session.session_id

    engine.session.message_history = [_user("q1"), _assistant("a1")]
    save_new_messages(engine)

    # Simulate /new
    engine.session.message_history.clear()
    engine.session.session_id = store.new_id()
    engine.session.saved_count = 0

    engine.session.message_history = [_user("q2")]
    save_new_messages(engine)

    sessions = store.list_sessions()
    assert len(sessions) == 2
    ids = {s.session_id for s in sessions}
    assert old_id in ids
    assert engine.session.session_id in ids


# ── compaction resets saved count ────────────────────────────────────


def test_compaction_reset_allows_resave() -> None:
    """After compaction resets saved_count, full history is re-persisted."""
    engine, store = _make_engine()
    engine.session.message_history = [_user("q1"), _assistant("a1")]
    save_new_messages(engine)

    # Simulate compaction: replace history, reset count.
    engine.session.message_history = [_user("[summary]"), _user("q2")]
    engine.session.saved_count = 0
    save_new_messages(engine)

    loaded = store.load_messages(engine.session.session_id)
    # Original 2 + compacted 2 = 4 (old rows not removed).
    assert len(loaded) == 4


# ── session context propagated ───────────────────────────────────────


def test_session_context_on_stored_rows() -> None:
    """Repo and model context appear in session listings."""
    engine, store = _make_engine()
    engine.session.message_history = [_user("q")]
    save_new_messages(engine)

    sessions = store.list_sessions(repo_owner="acme", repo_name="app")
    assert len(sessions) == 1
    assert sessions[0].session_label == "acme/app — main"
