"""Tests for /stats — current session, historical, and global.

Data-first: a shared fixture seeds two sessions with realistic
conversation data (tool calls, token usage, cost).  Tests verify
behaviours against that data through ``engine.run_task()``.
"""

from __future__ import annotations

import time

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.usage import RequestUsage

from rbtr.engine import Engine
from rbtr.sessions.serialise import (
    FailedAttempt,
    FailureKind,
    FragmentKind,
    HistoryRepair,
    IncidentOutcome,
    RecoveryStrategy,
)

from .conftest import _user, drain, output_texts

# ── Shared test data ─────────────────────────────────────────────────

_SONNET_USAGE = RequestUsage(
    input_tokens=1200, output_tokens=80, cache_read_tokens=600, cache_write_tokens=50
)
_FINAL_USAGE = RequestUsage(
    input_tokens=1500, output_tokens=200, cache_read_tokens=900, cache_write_tokens=60
)
_GPT_USAGE = RequestUsage(input_tokens=500, output_tokens=150)


def _seed_sessions(engine: Engine) -> None:
    """Seed two sessions into the store.

    **Current session** (claude/sonnet): user prompt, two tool calls
    (read_file, grep), final text response.  Total cost $0.015.

    **Historical session** (openai/gpt-4o): simple Q&A.
    Cost $0.005, label "testowner/testrepo — feature".
    """
    engine._sync_store_context()

    # Current session.
    engine.store.save_messages(
        engine.state.session_id,
        [
            _user("review the auth module"),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name="read_file", args={"path": "auth.py"}, tool_call_id="t1")
                ],
                usage=_SONNET_USAGE,
                model_name="claude/sonnet",
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="read_file", content="class Auth: ...", tool_call_id="t1"
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name="grep", args={"search": "password"}, tool_call_id="t2")
                ],
                usage=_SONNET_USAGE,
                model_name="claude/sonnet",
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name="grep", content="auth.py:42", tool_call_id="t2")]
            ),
            ModelResponse(
                parts=[TextPart(content="The auth module looks solid.")],
                usage=_FINAL_USAGE,
                model_name="claude/sonnet",
            ),
        ],
        cost=0.015,
    )

    # Historical session — different model and label.
    other_id = engine.store.new_id()
    engine.store.save_messages(
        other_id,
        [
            _user("explain dependency injection"),
            ModelResponse(
                parts=[TextPart(content="DI is a design pattern...")],
                usage=_GPT_USAGE,
                model_name="openai/gpt-4o",
            ),
        ],
        model_name="openai/gpt-4o",
        session_label="testowner/testrepo — feature",
        cost=0.005,
    )

    # Seed incidents into the current session.
    engine.store.save_incident(
        engine.state.session_id,
        FragmentKind.LLM_ATTEMPT_FAILED,
        FailedAttempt(
            turn_id="t-fail-1",
            failure_kind=FailureKind.HISTORY_FORMAT,
            strategy=RecoveryStrategy.DEMOTE_THINKING,
            outcome=IncidentOutcome.RECOVERED,
        ),
    )
    engine.store.save_incident(
        engine.state.session_id,
        FragmentKind.LLM_HISTORY_REPAIR,
        HistoryRepair(
            strategy=RecoveryStrategy.DEMOTE_THINKING,
            reason="cross_provider_retry",
        ),
    )


@pytest.fixture
def stats_engine(engine: Engine) -> Engine:
    """Engine with two seeded sessions and live usage state."""
    engine.state.model_name = "claude/sonnet"
    engine.state.session_label = "testowner/testrepo — main"
    engine.state.session_started_at = time.time() - 185
    engine.state.usage.context_window = 200_000
    engine.state.usage.context_window_known = True
    engine.state.usage.last_input_tokens = 24_000

    _seed_sessions(engine)
    drain(engine.events)
    return engine


def _other_session_id(engine: Engine) -> str:
    current = engine.state.session_id
    for s in engine.store.list_sessions():
        if s.session_id != current:
            return s.session_id
    raise AssertionError("expected two sessions in store")


# ── Tests ────────────────────────────────────────────────────────────


def test_current_session(stats_engine: Engine) -> None:
    """Current session shows model, messages, tokens, cost, tools, context."""
    stats_engine.run_task("command", "/stats")
    texts = output_texts(drain(stats_engine.events))
    combined = " ".join(texts)

    # Session header with elapsed time.
    assert "Session" in combined
    assert "3m" in combined
    assert "claude/sonnet" in combined

    # Token section.
    assert "Input" in combined
    assert "Output" in combined
    assert "Cache read" in combined
    assert "hit rate" in combined

    # Cost.
    assert "$" in combined

    # Tools.
    assert "read_file" in combined
    assert "grep" in combined
    assert "Tool calls (2)" in combined

    # Context % (live usage).
    assert "Context" in combined
    assert "200k" in combined


def test_historical_session(stats_engine: Engine) -> None:
    """Historical session shows DB data, no context % or elapsed time."""
    other_id = _other_session_id(stats_engine)
    stats_engine.run_task("command", f"/stats {other_id}")
    texts = output_texts(drain(stats_engine.events))
    combined = " ".join(texts)

    assert "testowner/testrepo" in combined
    assert "openai/gpt-4o" in combined
    assert "Input" in combined
    assert "$" in combined
    # No live data for historical sessions.
    assert "Context" not in combined


def test_historical_unknown_prefix(stats_engine: Engine) -> None:
    """Unknown prefix warns."""
    stats_engine.run_task("command", "/stats zzz-nonexistent")
    texts = output_texts(drain(stats_engine.events))
    assert any("No session" in t for t in texts)


def test_global_stats(stats_engine: Engine) -> None:
    """Global stats aggregates both sessions — both models, all tools."""
    stats_engine.run_task("command", "/stats all")
    texts = output_texts(drain(stats_engine.events))
    combined = " ".join(texts)

    assert "All sessions (2)" in combined
    assert "claude/sonnet" in combined
    assert "openai/gpt-4o" in combined
    assert "$" in combined
    assert "read_file" in combined
    assert "grep" in combined


def test_current_session_incidents(stats_engine: Engine) -> None:
    """/stats shows failures and repairs for the current session."""
    stats_engine.run_task("command", "/stats")
    texts = output_texts(drain(stats_engine.events))
    combined = " ".join(texts)

    assert "Failures" in combined
    assert "history_format" in combined
    assert "recovered" in combined
    assert "History repairs" in combined
    assert "demote_thinking" in combined
    assert "Recovery rate" in combined


def test_global_stats_incidents(stats_engine: Engine) -> None:
    """/stats all shows aggregated incident stats across all sessions."""
    stats_engine.run_task("command", "/stats all")
    texts = output_texts(drain(stats_engine.events))
    combined = " ".join(texts)

    assert "Failures" in combined
    assert "history_format" in combined
    assert "History repairs" in combined
    assert "demote_thinking" in combined
    assert "Recovery rate" in combined


def test_empty_session(engine: Engine) -> None:
    """Fresh session shows header and message count but no token/cost sections."""
    drain(engine.events)
    engine.run_task("command", "/stats")
    texts = output_texts(drain(engine.events))
    combined = " ".join(texts)

    assert "Session" in combined
    assert "Turns" in combined
    assert "Responses" in combined
    # No data → no token/cost/tool sections.
    assert "Input" not in combined
    assert "$" not in combined
    assert "Tool calls" not in combined


def test_global_empty(engine: Engine) -> None:
    """Global stats with no sessions shows a message."""
    drain(engine.events)
    engine.run_task("command", "/stats all")
    texts = output_texts(drain(engine.events))
    assert any("No sessions" in t for t in texts)
