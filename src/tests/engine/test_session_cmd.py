"""Tests for /session command — list, info, delete."""

from __future__ import annotations

import queue
from datetime import UTC, datetime, timedelta

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import RequestUsage

from rbtr.engine import Engine, EngineState
from rbtr.engine.save import save_new_messages
from rbtr.engine.session_cmd import _format_age, _parse_duration
from rbtr.events import Event

from .conftest import drain, make_engine, output_texts

_USAGE = RequestUsage(input_tokens=100, output_tokens=50)


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)], usage=_USAGE, model_name="test")


def _seed_engine() -> tuple[Engine, queue.Queue[Event], EngineState]:
    """Create an engine with two saved messages."""
    engine, events, session = make_engine()
    session.session_label = "testowner/testrepo — main"
    session.model_name = "claude/sonnet"
    session.message_history = [_user("hello"), _assistant("hi")]
    save_new_messages(engine, run_cost=0.01)
    drain(events)  # clear setup events
    return engine, events, session


# ── /session list ────────────────────────────────────────────────────


def test_list_shows_current_session() -> None:
    """List includes the active session with marker."""
    engine, events, _ = _seed_engine()
    engine.run_task("command", "/session list")
    texts = output_texts(drain(events))
    # Should have at least one line with the session short ID and marker.
    assert any("◂" in t for t in texts)


def test_list_empty_when_no_sessions() -> None:
    """List on a fresh store shows 'No sessions found'."""
    engine, events, _ = make_engine()
    drain(events)
    engine.run_task("command", "/session list --all")
    texts = output_texts(drain(events))
    assert any("No sessions" in t for t in texts)


def test_list_filters_by_repo() -> None:
    """Default list filters by current repo."""
    engine, events, session = _seed_engine()

    # Save messages under a different repo context.
    old_owner = session.owner
    session.owner = "other"
    session.repo_name = "other-repo"
    session.session_id = engine._store.new_id()
    session.saved_count = 0
    session.message_history = [_user("other")]
    save_new_messages(engine)

    # Restore original repo.
    session.owner = old_owner
    session.repo_name = "testrepo"

    drain(events)
    engine.run_task("command", "/session list")
    texts = output_texts(drain(events))
    # Should not include "other-repo" session.
    assert not any("other-repo" in t for t in texts)


def test_list_all_shows_all_repos() -> None:
    """--all shows sessions from all repos."""
    engine, events, session = _seed_engine()

    session.owner = "other"
    session.repo_name = "lib"
    session.session_label = "other/lib — main"
    session.session_id = engine._store.new_id()
    session.saved_count = 0
    session.message_history = [_user("from lib")]
    save_new_messages(engine)

    drain(events)
    engine.run_task("command", "/session list --all")
    texts = output_texts(drain(events))
    # Both sessions should appear.
    combined = " ".join(texts)
    assert "testowner/testrepo" in combined
    assert "other/lib" in combined


# ── /session info ────────────────────────────────────────────────────


def test_info_shows_session_details() -> None:
    """Info displays current session metadata."""
    engine, events, session = _seed_engine()
    engine.run_task("command", "/session info")
    texts = output_texts(drain(events))
    combined = " ".join(texts)
    assert session.session_id[:8] in combined
    assert "testowner/testrepo" in combined


# ── /session delete ──────────────────────────────────────────────────


def test_delete_by_prefix() -> None:
    """Delete a session by ID prefix."""
    engine, events, session = _seed_engine()

    # Create a second session to delete.
    old_id = session.session_id
    session.session_id = engine._store.new_id()
    session.saved_count = 0
    session.message_history = [_user("to delete")]
    save_new_messages(engine)
    target_id = session.session_id

    # Switch back to original.
    session.session_id = old_id

    drain(events)
    # Use full ID to avoid ambiguous prefix (UUID7s share timestamp prefix).
    engine.run_task("command", f"/session delete {target_id}")
    texts = output_texts(drain(events))
    assert any("Deleted" in t for t in texts)

    # Verify it's gone.
    sessions = engine._store.list_sessions()
    ids = {s.session_id for s in sessions}
    assert target_id not in ids


def test_delete_refuses_active_session() -> None:
    """Cannot delete the currently active session."""
    engine, events, session = _seed_engine()
    drain(events)
    engine.run_task("command", f"/session delete {session.session_id}")
    texts = output_texts(drain(events))
    assert any("active session" in t.lower() for t in texts)


def test_delete_unknown_prefix() -> None:
    """Deleting a non-existent prefix warns."""
    engine, events, _ = _seed_engine()
    drain(events)
    engine.run_task("command", "/session delete nonexistent")
    texts = output_texts(drain(events))
    assert any("No session" in t for t in texts)


def test_delete_before_duration() -> None:
    """Sessions newer than the cutoff are preserved."""
    engine, events, _ = _seed_engine()
    drain(events)
    # EngineState was just created — 999 days ago is far in the past, so nothing qualifies.
    engine.run_task("command", "/session delete --before 999d")
    texts = output_texts(drain(events))
    assert any("Deleted 0" in t for t in texts)


def test_delete_before_invalid_duration() -> None:
    """Invalid duration format warns."""
    engine, events, _ = _seed_engine()
    drain(events)
    engine.run_task("command", "/session delete --before xyz")
    texts = output_texts(drain(events))
    assert any("Invalid duration" in t for t in texts)


# ── _parse_duration ──────────────────────────────────────────────────


def test_parse_duration_days() -> None:
    d = _parse_duration("7d")
    assert d == timedelta(days=7)


def test_parse_duration_weeks() -> None:
    d = _parse_duration("2w")
    assert d == timedelta(weeks=2)


def test_parse_duration_hours() -> None:
    d = _parse_duration("24h")
    assert d == timedelta(hours=24)


def test_parse_duration_invalid() -> None:
    assert _parse_duration("abc") is None
    assert _parse_duration("") is None
    assert _parse_duration("7x") is None


# ── _format_age ──────────────────────────────────────────────────────


def test_format_age_now() -> None:
    ts = datetime.now(UTC).isoformat()
    assert _format_age(ts) in ("now", "1m")


def test_format_age_days() -> None:
    ts = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    assert _format_age(ts) == "3d"


def test_format_age_invalid() -> None:
    assert _format_age("not-a-date") == "?"


# ── config defaults ──────────────────────────────────────────────────


def test_sessions_config_defaults() -> None:
    """Sessions config has sensible defaults."""
    from rbtr.config import config

    assert config.sessions.max_sessions == 100
    assert config.sessions.max_age_days == 30
