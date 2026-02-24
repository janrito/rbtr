"""Tests for SessionStore CRUD — save, load, list, delete, search, compact."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from rbtr.sessions.serialise import MessageKind, MessageRow, SessionContext, prepare_row
from rbtr.sessions.store import SessionStore

# ── Shared data ──────────────────────────────────────────────────────

CTX_A = SessionContext(
    session_id="ses-aaa",
    session_label="acme/app — main",
    repo_owner="acme",
    repo_name="app",
    model_name="claude/sonnet",
)

CTX_B = SessionContext(
    session_id="ses-bbb",
    session_label="acme/lib — pr-5",
    repo_owner="acme",
    repo_name="lib",
    model_name="chatgpt/o3",
)

_USAGE = RequestUsage(input_tokens=100, output_tokens=50)


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)], usage=_USAGE, model_name="test")


def _make_rows(
    ctx: SessionContext,
    messages: list[ModelRequest | ModelResponse | str],
    *,
    start_id: int = 0,
) -> list[MessageRow]:
    """Build MessageRow list from messages."""
    rows = []
    for i, msg in enumerate(messages):
        rid = f"row-{ctx.session_id}-{start_id + i}"
        match msg:
            case str():
                kind = MessageKind.COMMAND if msg.startswith("/") else MessageKind.SHELL
                rows.append(prepare_row(msg, context=ctx, row_id=rid, kind=kind))
            case _:
                rows.append(prepare_row(msg, context=ctx, row_id=rid))
    return rows


# ═══════════════════════════════════════════════════════════════════════
# save_messages + load_messages
# ═══════════════════════════════════════════════════════════════════════


def test_save_and_load_roundtrip() -> None:
    """Save messages, load them back, verify lossless roundtrip."""
    msgs = [_user("hello"), _assistant("hi there")]
    rows = _make_rows(CTX_A, msgs)

    with SessionStore() as store:
        store.save_messages(rows)
        loaded = store.load_messages("ses-aaa")

    assert len(loaded) == 2
    assert loaded[0] == msgs[0]
    assert loaded[1] == msgs[1]


def test_save_empty_is_noop() -> None:
    """Saving an empty list doesn't error."""
    with SessionStore() as store:
        store.save_messages([])
        assert store.load_messages("nonexistent") == []


def test_append_only() -> None:
    """Saving more messages appends, doesn't replace."""
    batch_1 = _make_rows(CTX_A, [_user("q1"), _assistant("a1")])
    batch_2 = _make_rows(CTX_A, [_user("q2"), _assistant("a2")], start_id=2)

    with SessionStore() as store:
        store.save_messages(batch_1)
        store.save_messages(batch_2)
        loaded = store.load_messages("ses-aaa")

    assert len(loaded) == 4


def test_load_skips_command_and_shell_rows() -> None:
    """load_messages only returns LLM messages (has message_json)."""
    msgs: list[ModelRequest | ModelResponse | str] = [
        _user("hello"),
        "/review 42",
        _assistant("ok"),
        "!git status",
    ]
    rows = _make_rows(CTX_A, msgs)

    with SessionStore() as store:
        store.save_messages(rows)
        loaded = store.load_messages("ses-aaa")

    # Only the ModelRequest and ModelResponse are returned.
    assert len(loaded) == 2


def test_load_skips_corrupt_json() -> None:
    """Corrupt message_json is skipped, not raised."""
    rows = _make_rows(CTX_A, [_user("good")])

    with SessionStore() as store:
        store.save_messages(rows)
        # Inject a corrupt row directly.
        store._con.execute(
            "INSERT INTO messages (id, session_id, created_at, kind, message_json) "
            "VALUES ('bad', 'ses-aaa', '2099-01-01T00:00:00', 'response', '{corrupt}')"
        )
        store._con.commit()
        loaded = store.load_messages("ses-aaa")

    assert len(loaded) == 1  # only the good row


# ═══════════════════════════════════════════════════════════════════════
# list_sessions
# ═══════════════════════════════════════════════════════════════════════


def test_list_sessions_groups_by_session_id() -> None:
    """Each session appears once with correct counts."""
    rows_a = _make_rows(CTX_A, [_user("q1"), _assistant("a1")])
    rows_b = _make_rows(CTX_B, [_user("q2"), _assistant("a2"), _user("q3")])

    with SessionStore() as store:
        store.save_messages(rows_a)
        store.save_messages(rows_b)
        sessions = store.list_sessions()

    assert len(sessions) == 2
    by_id = {s.session_id: s for s in sessions}
    assert by_id["ses-aaa"].message_count == 2
    assert by_id["ses-bbb"].message_count == 3


def test_list_sessions_filter_by_repo() -> None:
    """Filtering by repo_owner and repo_name works."""
    rows_a = _make_rows(CTX_A, [_user("q1")])
    rows_b = _make_rows(CTX_B, [_user("q2")])

    with SessionStore() as store:
        store.save_messages(rows_a)
        store.save_messages(rows_b)
        filtered = store.list_sessions(repo_owner="acme", repo_name="lib")

    assert len(filtered) == 1
    assert filtered[0].session_id == "ses-bbb"


def test_list_sessions_respects_limit() -> None:
    """Limit caps the number of sessions returned."""
    rows_a = _make_rows(CTX_A, [_user("q1")])
    rows_b = _make_rows(CTX_B, [_user("q2")])

    with SessionStore() as store:
        store.save_messages(rows_a)
        store.save_messages(rows_b)
        sessions = store.list_sessions(limit=1)

    assert len(sessions) == 1


def test_list_sessions_ordered_by_recency() -> None:
    """Most recently active session comes first."""
    rows_a = _make_rows(CTX_A, [_user("old")])
    time.sleep(0.01)
    rows_b = _make_rows(CTX_B, [_user("new")])

    with SessionStore() as store:
        store.save_messages(rows_a)
        store.save_messages(rows_b)
        sessions = store.list_sessions()

    assert sessions[0].session_id == "ses-bbb"


# ═══════════════════════════════════════════════════════════════════════
# delete_session
# ═══════════════════════════════════════════════════════════════════════


def test_delete_session_removes_all_rows() -> None:
    """Deleting a session removes all its messages."""
    rows = _make_rows(CTX_A, [_user("q1"), _assistant("a1")])

    with SessionStore() as store:
        store.save_messages(rows)
        deleted = store.delete_session("ses-aaa")
        assert deleted == 2
        assert store.load_messages("ses-aaa") == []


def test_delete_session_leaves_others() -> None:
    """Deleting one session doesn't affect another."""
    rows_a = _make_rows(CTX_A, [_user("q1")])
    rows_b = _make_rows(CTX_B, [_user("q2")])

    with SessionStore() as store:
        store.save_messages(rows_a)
        store.save_messages(rows_b)
        store.delete_session("ses-aaa")
        assert store.load_messages("ses-bbb") != []


# ═══════════════════════════════════════════════════════════════════════
# delete_old_sessions
# ═══════════════════════════════════════════════════════════════════════


def test_delete_old_sessions() -> None:
    """Sessions older than cutoff are deleted."""
    rows = _make_rows(CTX_A, [_user("old")])

    with SessionStore() as store:
        store.save_messages(rows)
        # Delete anything before "the future".
        future = datetime(2099, 1, 1, tzinfo=UTC)
        deleted = store.delete_old_sessions(before=future)
        assert deleted > 0
        assert store.list_sessions() == []


def test_delete_old_sessions_keeps_recent() -> None:
    """Sessions newer than cutoff are kept."""
    rows = _make_rows(CTX_A, [_user("recent")])

    with SessionStore() as store:
        store.save_messages(rows)
        # Delete anything before "the past".
        past = datetime(2000, 1, 1, tzinfo=UTC)
        store.delete_old_sessions(before=past)
        assert len(store.list_sessions()) == 1


# ═══════════════════════════════════════════════════════════════════════
# search_history
# ═══════════════════════════════════════════════════════════════════════


def test_search_history_returns_user_text() -> None:
    """Search returns user prompts and commands."""
    msgs: list[ModelRequest | ModelResponse | str] = [
        _user("review tui.py"),
        _assistant("ok"),
        "/help",
    ]
    rows = _make_rows(CTX_A, msgs)

    with SessionStore() as store:
        store.save_messages(rows)
        results = store.search_history()

    assert "review tui.py" in results
    assert "/help" in results
    # Assistant text is not user_text.
    assert "ok" not in results


def test_search_history_prefix_filter() -> None:
    """Prefix search filters results."""
    rows = _make_rows(CTX_A, [_user("review file"), _user("explain code")])

    with SessionStore() as store:
        store.save_messages(rows)
        results = store.search_history(prefix="review")

    assert len(results) == 1
    assert results[0] == "review file"


def test_search_history_cross_session() -> None:
    """History spans all sessions."""
    rows_a = _make_rows(CTX_A, [_user("from session A")])
    rows_b = _make_rows(CTX_B, [_user("from session B")])

    with SessionStore() as store:
        store.save_messages(rows_a)
        store.save_messages(rows_b)
        results = store.search_history()

    assert len(results) == 2


def test_search_history_respects_limit() -> None:
    """Limit caps results."""
    rows = _make_rows(CTX_A, [_user(f"q{i}") for i in range(10)])

    with SessionStore() as store:
        store.save_messages(rows)
        results = store.search_history(limit=3)

    assert len(results) == 3


# ═══════════════════════════════════════════════════════════════════════
# mark_compacted
# ═══════════════════════════════════════════════════════════════════════


def test_mark_session_compacted_hides_from_load() -> None:
    """All rows for a session are marked, then excluded from load."""
    msgs = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
    rows = _make_rows(CTX_A, msgs)

    with SessionStore() as store:
        store.save_messages(rows)
        marked = store.mark_session_compacted("ses-aaa", summary_id="summary-001")
        assert marked == 4

        loaded = store.load_messages("ses-aaa")

    assert len(loaded) == 0


def test_mark_session_compacted_does_not_affect_other_sessions() -> None:
    """Marking one session leaves other sessions intact."""
    rows_a = _make_rows(CTX_A, [_user("q1")])
    rows_b = _make_rows(CTX_B, [_user("q2")])

    with SessionStore() as store:
        store.save_messages(rows_a)
        store.save_messages(rows_b)
        store.mark_session_compacted("ses-aaa", summary_id="s1")

        loaded_a = store.load_messages("ses-aaa")
        loaded_b = store.load_messages("ses-bbb")

    assert len(loaded_a) == 0
    assert len(loaded_b) == 1


# ═══════════════════════════════════════════════════════════════════════
# delete_excess_sessions
# ═══════════════════════════════════════════════════════════════════════


def test_delete_excess_keeps_recent() -> None:
    """Keeps the N most recent sessions, deletes the rest."""
    rows_a = _make_rows(CTX_A, [_user("old")])
    time.sleep(0.01)
    rows_b = _make_rows(CTX_B, [_user("new")])

    with SessionStore() as store:
        store.save_messages(rows_a)
        store.save_messages(rows_b)
        deleted = store.delete_excess_sessions(keep=1)
        assert deleted > 0
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == "ses-bbb"


def test_delete_excess_noop_when_under_limit() -> None:
    """No deletion when session count is at or below the limit."""
    rows = _make_rows(CTX_A, [_user("only")])

    with SessionStore() as store:
        store.save_messages(rows)
        deleted = store.delete_excess_sessions(keep=5)
        assert deleted == 0
        assert len(store.list_sessions()) == 1


def test_delete_excess_zero_keep_is_noop() -> None:
    """keep=0 or negative doesn't delete anything (safety guard)."""
    rows = _make_rows(CTX_A, [_user("safe")])

    with SessionStore() as store:
        store.save_messages(rows)
        assert store.delete_excess_sessions(keep=0) == 0
        assert len(store.list_sessions()) == 1
