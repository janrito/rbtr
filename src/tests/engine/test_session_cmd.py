"""Tests for /session command — list, info, delete, purge, resume."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse
from pytest_mock import MockerFixture

from rbtr.engine import Engine
from rbtr.engine.session_cmd import _format_age, _parse_duration
from rbtr.providers import BuiltinProvider

from .conftest import _assistant, _seed, _user, drain, output_texts


@pytest.fixture
def seeded_engine(engine: Engine) -> Engine:
    """Engine with two saved messages."""
    engine.state.session_label = "testowner/testrepo — main"
    engine.state.model_name = "claude/sonnet"
    _seed(engine, [_user("hello"), _assistant("hi")], cost=0.01)
    drain(engine.events)
    return engine


@pytest.fixture
def double_seeded_engine(seeded_engine: Engine) -> Engine:
    """Seeded engine with a second session saved to the store.

    On return the engine is back on the *first* session.
    The second session's ID is discoverable via ``list_sessions``.
    """
    engine = seeded_engine
    first_id = engine.state.session_id

    engine.state.session_id = engine.store.new_id()
    _seed(engine, [_user("second"), _assistant("reply")], cost=0.02)

    # Switch back to first.
    engine.state.session_id = first_id

    drain(engine.events)
    return engine


def _other_session_id(engine: Engine) -> str:
    """Return the session ID that is *not* the current one."""
    current = engine.state.session_id
    for s in engine.store.list_sessions():
        if s.session_id != current:
            return s.session_id
    raise AssertionError("expected two sessions in store")


# ── /session (list) ─────────────────────────────────────────────────


def test_list_shows_current_session(seeded_engine: Engine) -> None:
    """Default list includes the active session with marker."""
    engine = seeded_engine
    engine.run_task("command", "/session")
    texts = output_texts(drain(engine.events))
    assert any("◂" in t for t in texts)


def test_list_empty_when_no_sessions(engine: Engine) -> None:
    drain(engine.events)
    engine.run_task("command", "/session")
    texts = output_texts(drain(engine.events))
    assert any("No sessions" in t for t in texts)


def test_list_filters_by_repo(seeded_engine: Engine) -> None:
    """Default list filters by current repo."""
    engine = seeded_engine

    # Save messages under a different repo context.
    old_owner = engine.state.owner
    old_repo = engine.state.repo_name
    engine.state.owner = "other"
    engine.state.repo_name = "other-repo"
    engine.state.session_id = engine.store.new_id()
    engine.store.save_messages(
        engine.state.session_id,
        [_user("other")],
        repo_owner="other",
        repo_name="other-repo",
    )

    engine.state.owner = old_owner
    engine.state.repo_name = old_repo

    drain(engine.events)
    engine.run_task("command", "/session")
    texts = output_texts(drain(engine.events))
    assert not any("other-repo" in t for t in texts)


# ── /session all ─────────────────────────────────────────────────────


def test_all_shows_all_repos(seeded_engine: Engine) -> None:
    """all shows sessions from all repos."""
    engine = seeded_engine

    engine.state.session_id = engine.store.new_id()
    engine.store.save_messages(
        engine.state.session_id,
        [_user("from lib")],
        session_label="other/lib — main",
        repo_owner="other",
        repo_name="lib",
    )

    drain(engine.events)
    engine.run_task("command", "/session all")
    texts = output_texts(drain(engine.events))
    combined = " ".join(texts)
    assert "testowner/testrepo" in combined
    assert "other/lib" in combined


# ── /session info ────────────────────────────────────────────────────


def test_info_shows_session_details(seeded_engine: Engine) -> None:
    engine = seeded_engine
    engine.run_task("command", "/session info")
    texts = output_texts(drain(engine.events))
    combined = " ".join(texts)
    assert engine.state.session_id[:8] in combined
    assert "testowner/testrepo" in combined


# ── /session resume ──────────────────────────────────────────────────


def test_resume_loads_messages(double_seeded_engine: Engine) -> None:
    """Resume restores messages, session ID, and usage from DB."""
    engine = double_seeded_engine
    second_id = _other_session_id(engine)

    engine.run_task("command", f"/session resume {second_id}")
    texts = output_texts(drain(engine.events))
    assert any("Resumed" in t for t in texts)
    assert engine.state.session_id == second_id
    assert len(engine.store.load_messages(second_id)) == 2

    # Usage restored from DB stats — turn/response counts and cost
    # reflect the resumed session's history.
    ts = engine.store.token_stats(second_id)
    assert engine.state.usage.turn_count == ts.total_turns
    assert engine.state.usage.response_count == ts.total_responses
    assert engine.state.usage.total_cost == ts.total_cost


def test_resume_already_active(seeded_engine: Engine) -> None:
    """Resuming the current session warns."""
    engine = seeded_engine
    engine.run_task("command", f"/session resume {engine.state.session_id}")
    texts = output_texts(drain(engine.events))
    assert any("Already" in t for t in texts)


def test_resume_unknown_prefix(seeded_engine: Engine) -> None:
    engine = seeded_engine
    engine.run_task("command", "/session resume nonexistent")
    texts = output_texts(drain(engine.events))
    assert any("No session" in t for t in texts)


def test_resume_no_args(seeded_engine: Engine) -> None:
    engine = seeded_engine
    engine.run_task("command", "/session resume")
    texts = output_texts(drain(engine.events))
    assert any("Usage" in t for t in texts)


def test_resume_after_compaction(mocker: MockerFixture, engine: Engine) -> None:
    """Resume after compaction loads only the post-compaction state."""
    mocker.patch(
        "rbtr.llm.compact._stream_summary",
        return_value="Summary of conversation.",
    )
    mocker.patch("rbtr.llm.compact.build_model")

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/sonnet"
    engine.state.session_label = "test/repo — main"
    engine.state.usage.context_window = 200_000

    # Build 15 turns and save.
    history: list[ModelRequest | ModelResponse] = []
    for i in range(15):
        history.extend([_user(f"q{i}"), _assistant(f"a{i}")])
    _seed(engine, history)
    compacted_session_id = engine.state.session_id

    # Compact (rewrites DB).
    from rbtr.llm.compact import compact_history

    compact_history(engine._llm_context())
    post_compact_count = len(engine.store.load_messages(compacted_session_id))

    # Switch to a new session.
    engine.state.session_id = engine.store.new_id()
    _seed(engine, [_user("new")])

    drain(engine.events)

    # Resume the compacted session.
    engine.run_task("command", f"/session resume {compacted_session_id}")
    texts = output_texts(drain(engine.events))
    assert any("Resumed" in t for t in texts)
    assert engine.state.session_id == compacted_session_id
    assert len(engine.store.load_messages(compacted_session_id)) == post_compact_count


# ── /session delete ──────────────────────────────────────────────────


def test_delete_by_prefix(double_seeded_engine: Engine) -> None:
    engine = double_seeded_engine
    second_id = _other_session_id(engine)

    engine.run_task("command", f"/session delete {second_id}")
    texts = output_texts(drain(engine.events))
    assert any("Deleted" in t for t in texts)

    ids = {s.session_id for s in engine.store.list_sessions()}
    assert second_id not in ids


def test_delete_refuses_active_session(seeded_engine: Engine) -> None:
    engine = seeded_engine
    engine.run_task("command", f"/session delete {engine.state.session_id}")
    texts = output_texts(drain(engine.events))
    assert any("active session" in t.lower() for t in texts)


def test_delete_unknown_prefix(seeded_engine: Engine) -> None:
    engine = seeded_engine
    engine.run_task("command", "/session delete nonexistent")
    texts = output_texts(drain(engine.events))
    assert any("No session" in t for t in texts)


# ── /session purge ───────────────────────────────────────────────────


def test_purge_nothing_recent(seeded_engine: Engine) -> None:
    """Sessions newer than the cutoff are preserved."""
    engine = seeded_engine
    engine.run_task("command", "/session purge 999d")
    texts = output_texts(drain(engine.events))
    assert any("Deleted 0" in t for t in texts)


def test_purge_invalid_duration(seeded_engine: Engine) -> None:
    engine = seeded_engine
    engine.run_task("command", "/session purge xyz")
    texts = output_texts(drain(engine.events))
    assert any("Invalid duration" in t for t in texts)


def test_purge_no_args(seeded_engine: Engine) -> None:
    engine = seeded_engine
    engine.run_task("command", "/session purge")
    texts = output_texts(drain(engine.events))
    assert any("Usage" in t for t in texts)


# ── _parse_duration ──────────────────────────────────────────────────


def test_parse_duration_days() -> None:
    assert _parse_duration("7d") == timedelta(days=7)


def test_parse_duration_weeks() -> None:
    assert _parse_duration("2w") == timedelta(weeks=2)


def test_parse_duration_hours() -> None:
    assert _parse_duration("24h") == timedelta(hours=24)


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
