"""Tests for SessionStore — the public API for session persistence.

Data-first: realistic conversation data is defined once as module
constants.  Tests verify **behaviours** through the store's public
API (save, load, list, search, compact, delete, streaming) — not
internal serialisation functions.

Organisation:
- Shared data
- Round-trip: save → load preserves messages losslessly
- Streaming: begin_response → add_part → finish → load
- Listing & search
- Compaction
- Deletion & pruning
- Schema & lifecycle
- FK cascade & compaction integrity
"""

from __future__ import annotations

import base64
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage
from uuid_utils import uuid7

from rbtr.sessions.store import _SCHEMA_VERSION, SessionStore

# ═══════════════════════════════════════════════════════════════════════
# Shared test data
# ═══════════════════════════════════════════════════════════════════════

_USAGE = RequestUsage(input_tokens=100, output_tokens=50)
_USAGE_LARGE = RequestUsage(input_tokens=5000, output_tokens=200)


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _system(text: str) -> ModelRequest:
    return ModelRequest(parts=[SystemPromptPart(content=text)])


def _assistant(text: str, *, model: str = "test-model") -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)], usage=_USAGE, model_name=model)


def _tool_call_response(name: str, args: dict[str, str]) -> ModelResponse:
    return ModelResponse(
        parts=[
            TextPart(content="Let me check."),
            ToolCallPart(tool_name=name, args=args, tool_call_id="tc1"),
        ],
        usage=_USAGE_LARGE,
        model_name="test-model",
    )


def _tool_return_request(name: str, content: str) -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name=name, content=content, tool_call_id="tc1")])


def _thinking_response(thinking: str, text: str) -> ModelResponse:
    return ModelResponse(
        parts=[ThinkingPart(content=thinking), TextPart(content=text)],
        usage=_USAGE_LARGE,
        model_name="test-model",
    )


# A realistic multi-turn conversation with tool calls and thinking.
CONVERSATION: list[ModelRequest | ModelResponse] = [
    _user("review src/tui.py"),
    _assistant("I'll read the file."),
    _tool_call_response("read_file", {"path": "src/tui.py"}),
    _tool_return_request("read_file", "class TUI:\n    pass"),
    _thinking_response("The TUI class is minimal...", "Here's my analysis."),
    _user("any security issues?"),
    _assistant("No security issues found."),
]

# Tool return request for standalone tests.
TOOL_RETURN = _tool_return_request("read_file", "def hello(): ...")

# Multi-tool response.
MULTI_TOOL_RESPONSE = ModelResponse(
    parts=[
        TextPart(content="Let me check."),
        ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
        ToolCallPart(tool_name="grep", args={"pattern": "TODO"}, tool_call_id="tc2"),
    ],
    usage=_USAGE_LARGE,
    model_name="claude-sonnet-4-20250514",
)


# ═══════════════════════════════════════════════════════════════════════
# Save → load round-trips
# ═══════════════════════════════════════════════════════════════════════


def test_roundtrip_full_conversation() -> None:
    """The full realistic conversation round-trips losslessly."""
    with SessionStore() as store:
        store.save_messages("s1", CONVERSATION)
        loaded = store.load_messages("s1")

    assert len(loaded) == len(CONVERSATION)
    for original, restored in zip(CONVERSATION, loaded, strict=True):
        assert restored == original


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("user_prompt", _user("explain closures")),
        ("system_prompt", _system("be helpful")),
        ("text_response", _assistant("A closure captures variables.")),
        ("tool_call", _tool_call_response("read_file", {"path": "a.py"})),
        ("tool_return", TOOL_RETURN),
        ("thinking", _thinking_response("hmm...", "Here's my answer.")),
        ("multi_tool", MULTI_TOOL_RESPONSE),
        (
            "retry",
            ModelRequest(
                parts=[
                    RetryPromptPart(content="try again", tool_name="f", tool_call_id="tc1"),
                ]
            ),
        ),
        (
            "zero_usage",
            ModelResponse(
                parts=[TextPart(content="hi")],
                model_name="test",
            ),
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_roundtrip_single_message(label: str, message: ModelRequest | ModelResponse) -> None:
    """Each message type round-trips individually."""
    with SessionStore() as store:
        store.save_messages("s1", [message])
        loaded = store.load_messages("s1")

    assert len(loaded) == 1
    assert loaded[0] == message


def test_incremental_saves() -> None:
    """Multiple save_messages calls accumulate."""
    with SessionStore() as store:
        store.save_messages("s1", CONVERSATION[:2])
        store.save_messages("s1", CONVERSATION[2:])
        loaded = store.load_messages("s1")

    assert len(loaded) == len(CONVERSATION)


def test_save_empty_is_noop() -> None:
    """Saving an empty list doesn't error."""
    with SessionStore() as store:
        store.save_messages("s1", [])
        assert store.load_messages("s1") == []


def test_load_unknown_session() -> None:
    """Loading a nonexistent session returns an empty list."""
    with SessionStore() as store:
        assert store.load_messages("nonexistent") == []


def test_sessions_dont_mix() -> None:
    """Messages saved to different sessions are isolated."""
    with SessionStore() as store:
        store.save_messages("s1", [_user("q1")])
        store.save_messages("s2", CONVERSATION[:2])
        assert len(store.load_messages("s1")) == 1
        assert len(store.load_messages("s2")) == 2


def test_command_shell_rows_excluded() -> None:
    """load_messages only returns LLM messages — command/shell are excluded."""
    with SessionStore() as store:
        store.save_messages("s1", [_user("hello"), _assistant("ok")])
        store.save_input("s1", "/review 42", "command")
        store.save_input("s1", "!git status", "shell")
        loaded = store.load_messages("s1")

    assert len(loaded) == 2


def test_corrupt_json_skipped() -> None:
    """Corrupt data_json is skipped, not raised."""
    with SessionStore() as store:
        store.save_messages("s1", [_user("good")])
        store._con.execute(
            "INSERT INTO fragments (id, session_id, message_id, fragment_index, fragment_kind, "
            "created_at, data_json, complete) "
            "VALUES ('bad', 's1', 'bad', 0, 'response-message', "
            "'2099-01-01T00:00:00', '{corrupt}', 1)"
        )
        store._con.commit()
        loaded = store.load_messages("s1")

    assert len(loaded) == 1


# ═══════════════════════════════════════════════════════════════════════
# Streaming: begin_response → add_part → finish_part → finish
# ═══════════════════════════════════════════════════════════════════════


def test_streaming_roundtrip() -> None:
    """Parts added via ResponseWriter are visible after finish."""
    with SessionStore() as store:
        store.set_context("s1")

        # Save the request first.
        store.save_messages("s1", [_user("hello")])

        # Stream a response.
        writer = store.begin_response("s1", model_name="test-model")
        part = TextPart(content="")
        writer.add_part(0, part)
        final_part = TextPart(content="Hello back!")
        writer.finish_part(0, final_part)
        writer.finish()

        loaded = store.load_messages("s1")

    assert len(loaded) == 2
    response = loaded[1]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "Hello back!"


def test_streaming_invisible_before_finish() -> None:
    """Incomplete response is not visible to load_messages."""
    with SessionStore() as store:
        store.set_context("s1")
        store.save_messages("s1", [_user("hello")])

        writer = store.begin_response("s1", model_name="test-model")
        writer.add_part(0, TextPart(content="partial"))
        # Do NOT call finish.

        loaded = store.load_messages("s1")

    # Only the request is visible.
    assert len(loaded) == 1


def test_streaming_with_cost() -> None:
    """Cost can be set when finishing or re-finishing."""
    with SessionStore() as store:
        store.set_context("s1")
        store.save_messages("s1", [_user("hello")])

        writer = store.begin_response("s1", model_name="test-model")
        writer.add_part(0, TextPart(content=""))
        writer.finish_part(0, TextPart(content="response"))
        writer.finish()  # without cost

        # Later, set cost (idempotent re-finish).
        writer.finish(cost=0.003)

        sessions = store.list_sessions()

    assert len(sessions) == 1
    assert sessions[0].total_cost == pytest.approx(0.003)


def test_streaming_context_manager() -> None:
    """ResponseWriter as context manager auto-finishes."""
    with SessionStore() as store:
        store.set_context("s1")
        store.save_messages("s1", [_user("hello")])

        with store.begin_response("s1", model_name="test") as writer:
            writer.add_part(0, TextPart(content=""))
            writer.finish_part(0, TextPart(content="done"))

        loaded = store.load_messages("s1")

    assert len(loaded) == 2


def test_streaming_multiple_parts() -> None:
    """Multiple parts are preserved in order."""
    with SessionStore() as store:
        store.set_context("s1")
        store.save_messages("s1", [_user("hello")])

        writer = store.begin_response("s1", model_name="test")
        writer.add_part(0, ThinkingPart(content=""))
        writer.add_part(1, TextPart(content=""))
        writer.finish_part(0, ThinkingPart(content="Let me think..."))
        writer.finish_part(1, TextPart(content="Here's my answer."))
        writer.finish()

        loaded = store.load_messages("s1")

    response = loaded[1]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 2
    assert isinstance(response.parts[0], ThinkingPart)
    assert isinstance(response.parts[1], TextPart)


# ═══════════════════════════════════════════════════════════════════════
# save_input: command/shell persistence
# ═══════════════════════════════════════════════════════════════════════


def test_save_input_excluded_from_load() -> None:
    """Command/shell inputs are excluded from load_messages."""
    with SessionStore() as store:
        store.save_messages("s1", [_user("hello"), _assistant("ok")])
        store.save_input("s1", "/compact", "command")
        store.save_input("s1", "!git status", "shell")
        loaded = store.load_messages("s1")

    assert len(loaded) == 2


def test_save_input_visible_in_search() -> None:
    """Command/shell inputs are found by search_history."""
    with SessionStore() as store:
        store.save_input("s1", "/review 42", "command")
        store.save_input("s1", "!ls -la", "shell")
        results = store.search_history()

    assert "/review 42" in results
    assert "!ls -la" in results


def test_save_input_prefix_search() -> None:
    """search_history prefix filter works on command/shell inputs."""
    with SessionStore() as store:
        store.save_input("s1", "/review 42", "command")
        store.save_input("s1", "/compact", "command")
        store.save_input("s1", "!git log", "shell")

        assert store.search_history(prefix="/review") == ["/review 42"]
        assert store.search_history(prefix="!git") == ["!git log"]


# ═══════════════════════════════════════════════════════════════════════
# Session listing
# ═══════════════════════════════════════════════════════════════════════


def test_list_sessions_groups_by_id() -> None:
    """Each session appears once with correct counts."""
    with SessionStore() as store:
        store.save_messages("s1", CONVERSATION[:2], repo_owner="acme", repo_name="app")
        store.save_messages("s2", CONVERSATION[:4], repo_owner="acme", repo_name="lib")
        sessions = store.list_sessions()

    assert len(sessions) == 2
    by_id = {s.session_id: s for s in sessions}
    assert by_id["s1"].message_count == 2
    assert by_id["s2"].message_count == 4


def test_list_sessions_filter_by_repo() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("q1")], repo_owner="acme", repo_name="app")
        store.save_messages("s2", [_user("q2")], repo_owner="acme", repo_name="lib")
        filtered = store.list_sessions(repo_owner="acme", repo_name="lib")

    assert len(filtered) == 1
    assert filtered[0].session_id == "s2"


def test_list_sessions_respects_limit() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("q1")])
        store.save_messages("s2", [_user("q2")])
        assert len(store.list_sessions(limit=1)) == 1


def test_list_sessions_ordered_by_recency() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("old")])
        time.sleep(0.01)
        store.save_messages("s2", [_user("new")])
        sessions = store.list_sessions()

    assert sessions[0].session_id == "s2"


def test_cost_in_listing() -> None:
    """run_cost appears in session listing."""
    with SessionStore() as store:
        store.save_messages("s1", [_user("q"), _assistant("a")], cost=0.05, model_name="claude")
        sessions = store.list_sessions()

    assert sessions[0].total_cost == pytest.approx(0.05)
    assert sessions[0].model_name == "claude"


# ═══════════════════════════════════════════════════════════════════════
# History search
# ═══════════════════════════════════════════════════════════════════════


def test_search_returns_user_prompts() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("review tui.py"), _assistant("ok")])
        results = store.search_history()

    assert "review tui.py" in results
    assert "ok" not in results  # assistant text excluded


def test_search_prefix_filter() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("review file"), _user("explain code")])
        results = store.search_history(prefix="review")

    assert results == ["review file"]


def test_search_cross_session() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("from A")])
        store.save_messages("s2", [_user("from B")])
        assert len(store.search_history()) == 2


def test_search_respects_limit() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user(f"q{i}") for i in range(10)])
        assert len(store.search_history(limit=3)) == 3


# ═══════════════════════════════════════════════════════════════════════
# Compaction
# ═══════════════════════════════════════════════════════════════════════


def test_compact_hides_old_messages() -> None:
    """Compacted messages disappear; summary + kept remain."""
    msgs = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]

    with SessionStore() as store:
        store.save_messages("s1", msgs)
        all_ids = store.load_message_ids("s1")
        store.compact_session("s1", summary=_user("[Summary]"), compact_ids=all_ids[:2])
        loaded = store.load_messages("s1")

    assert len(loaded) == 3  # summary + 2 kept
    assert loaded[0].parts[0].content == "[Summary]"  # type: ignore[union-attr]


def test_compact_then_save_more() -> None:
    """New messages after compaction are visible."""
    with SessionStore() as store:
        store.save_messages("s1", [_user("q1"), _assistant("a1")])
        all_ids = store.load_message_ids("s1")
        store.compact_session("s1", summary=_user("[summary]"), compact_ids=all_ids)

        store.save_messages("s1", [_user("q2"), _assistant("a2")])
        loaded = store.load_messages("s1")

    assert len(loaded) == 3  # summary + q2 + a2


@pytest.mark.parametrize("compact_count", [5, 4, 2])
def test_compact_varying_counts(compact_count: int) -> None:
    with SessionStore() as store:
        msgs = [_user(f"q{i}") for i in range(5)]
        store.save_messages("s1", msgs)
        all_ids = store.load_message_ids("s1")
        store.compact_session("s1", summary=_user("[summary]"), compact_ids=all_ids[:compact_count])
        loaded = store.load_messages("s1")

    assert len(loaded) == 1 + (5 - compact_count)


def test_compact_does_not_affect_other_sessions() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("q1")])
        store.save_messages("s2", [_user("q2")])
        all_ids = store.load_message_ids("s1")
        store.compact_session("s1", summary=_user("[summary]"), compact_ids=all_ids)

        assert len(store.load_messages("s1")) == 1  # summary only
        assert len(store.load_messages("s2")) == 1  # untouched


# ═══════════════════════════════════════════════════════════════════════
# Deletion & pruning
# ═══════════════════════════════════════════════════════════════════════


def test_delete_session() -> None:
    with SessionStore() as store:
        store.save_messages("s1", CONVERSATION[:2])
        deleted = store.delete_session("s1")
        assert deleted > 0
        assert store.load_messages("s1") == []


def test_delete_leaves_others() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("q1")])
        store.save_messages("s2", [_user("q2")])
        store.delete_session("s1")
        assert store.load_messages("s2") != []


def test_delete_old_sessions() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("old")])
        deleted = store.delete_old_sessions(before=datetime(2099, 1, 1, tzinfo=UTC))
        assert deleted > 0
        assert store.list_sessions() == []


def test_delete_old_keeps_recent() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("recent")])
        store.delete_old_sessions(before=datetime(2000, 1, 1, tzinfo=UTC))
        assert len(store.list_sessions()) == 1


def test_delete_excess_keeps_recent() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("old")])
        time.sleep(0.01)
        store.save_messages("s2", [_user("new")])
        deleted = store.delete_excess_sessions(keep=1)
        assert deleted > 0
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == "s2"


def test_delete_excess_noop_under_limit() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("only")])
        assert store.delete_excess_sessions(keep=5) == 0


def test_delete_excess_zero_is_noop() -> None:
    with SessionStore() as store:
        store.save_messages("s1", [_user("safe")])
        assert store.delete_excess_sessions(keep=0) == 0


# ═══════════════════════════════════════════════════════════════════════
# Schema & lifecycle
# ═══════════════════════════════════════════════════════════════════════


def test_schema_created() -> None:
    with SessionStore() as store:
        cur = store._con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fragments'"
        )
        assert cur.fetchone() is not None


def test_user_version_set() -> None:
    with SessionStore() as store:
        assert store._user_version() == _SCHEMA_VERSION


def test_wal_mode() -> None:
    with SessionStore() as store:
        row = store._con.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        assert row[0] in ("wal", "memory")


def test_indexes_created() -> None:
    with SessionStore() as store:
        cur = store._con.execute("SELECT name FROM sqlite_master WHERE type='index'")
        names = {row[0] for row in cur.fetchall()}
        assert "idx_fragments_message_fragment" in names
        assert "idx_fragments_session_created" in names
        assert "idx_fragments_user_text" in names
        assert "idx_fragments_compacted_by" in names


def test_new_id_is_uuid7() -> None:
    with SessionStore() as store:
        parsed = uuid.UUID(store.new_id())
        assert parsed.version == 7


def test_uuid7_sortability() -> None:
    first = uuid7()
    time.sleep(0.002)
    second = uuid7()
    assert str(first) < str(second)


def test_idempotent_open(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with SessionStore(db_path) as store:
        store.save_messages("s1", [_user("hello")])

    with SessionStore(db_path) as store:
        assert len(store.load_messages("s1")) == 1
        assert store._user_version() == _SCHEMA_VERSION


def test_newer_version_warns(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with SessionStore(db_path) as store:
        store._set_user_version(_SCHEMA_VERSION + 99)

    with SessionStore(db_path) as store:
        assert store._user_version() == _SCHEMA_VERSION + 99


# ═══════════════════════════════════════════════════════════════════════
# Round-trip: additional part types (FilePart, Builtin*)
# ═══════════════════════════════════════════════════════════════════════

# A tiny 1x1 red PNG for FilePart tests.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "2mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.mark.parametrize(
    ("label", "message"),
    [
        (
            "file_image",
            ModelResponse(
                parts=[
                    FilePart(content=BinaryContent(data=_TINY_PNG, media_type="image/png")),
                    TextPart(content="A 1x1 red pixel."),
                ],
                usage=_USAGE,
                model_name="test",
            ),
        ),
        (
            "builtin_tool_call",
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name="web_search",
                        args={"query": "weather today"},
                        tool_call_id="bt1",
                    ),
                ],
                usage=_USAGE,
                model_name="test",
            ),
        ),
        (
            "builtin_tool_return",
            ModelResponse(
                parts=[
                    BuiltinToolReturnPart(
                        tool_name="web_search",
                        content="Sunny, 72°F",
                        tool_call_id="bt1",
                    ),
                ],
                usage=_USAGE,
                model_name="test",
            ),
        ),
        (
            "mixed_response_file_and_text",
            ModelResponse(
                parts=[
                    ThinkingPart(content="analysing the image"),
                    FilePart(content=BinaryContent(data=_TINY_PNG, media_type="image/png")),
                    TextPart(content="review complete"),
                ],
                usage=_USAGE,
                model_name="test",
            ),
        ),
        (
            "mixed_response_thinking_tool_text",
            ModelResponse(
                parts=[
                    ThinkingPart(content="need to search first"),
                    ToolCallPart(tool_name="grep", args={"pattern": "TODO"}, tool_call_id="tc1"),
                    TextPart(content="let me check"),
                ],
                usage=_USAGE_LARGE,
                model_name="test",
            ),
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_roundtrip_extended_part_types(label: str, message: ModelRequest | ModelResponse) -> None:
    """Part types beyond the basic set round-trip correctly.

    FilePart with BinaryContent deserialises to BinaryImage (a subclass) —
    PydanticAI's after-validator narrows the type.  For those cases we
    compare data/media_type rather than strict equality.
    """
    with SessionStore() as store:
        store.save_messages("s1", [message])
        loaded = store.load_messages("s1")

    assert len(loaded) == 1

    original = message
    restored = loaded[0]

    # Compare part-by-part to handle BinaryContent → BinaryImage narrowing.
    assert len(restored.parts) == len(original.parts)
    for orig_part, rest_part in zip(original.parts, restored.parts, strict=True):
        if isinstance(orig_part, FilePart) and isinstance(rest_part, FilePart):
            assert rest_part.content.data == orig_part.content.data  # type: ignore[union-attr]
            assert rest_part.content.media_type == orig_part.content.media_type  # type: ignore[union-attr]
        else:
            assert rest_part == orig_part


# ═══════════════════════════════════════════════════════════════════════
# FK cascade & compaction integrity
# ═══════════════════════════════════════════════════════════════════════


def test_delete_session_cascades_through_compacted_rows() -> None:
    """Deleting a session removes compacted rows, summaries, and kept
    messages — no orphans left.
    """
    with SessionStore() as store:
        # 6 messages: 3 turns of Q/A.
        msgs = [_user(f"q{i}") for i in range(3)] + [_assistant(f"a{i}") for i in range(3)]
        store.save_messages("s1", msgs)

        # Compact first 4 messages (turns 1 & 2).
        all_ids = store.load_message_ids("s1")
        store.compact_session("s1", summary=_user("[summary]"), compact_ids=all_ids[:4])

        # Verify compacted state before delete.
        loaded = store.load_messages("s1")
        assert len(loaded) == 3  # summary + 2 kept

        # Delete the entire session.
        deleted = store.delete_session("s1")
        assert deleted > 0

        # Everything is gone — no orphans from message_id or compacted_by FKs.
        row = store._con.execute(
            "SELECT COUNT(*) AS cnt FROM fragments WHERE session_id = 's1'"
        ).fetchone()
        assert row is not None
        assert row["cnt"] == 0


def test_compaction_cascade_delete_summary_removes_marked_rows() -> None:
    """When a summary row is deleted, its compacted_by children are
    cascade-deleted too (FK ON DELETE CASCADE on compacted_by).
    """
    with SessionStore() as store:
        msgs = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
        store.save_messages("s1", msgs)

        all_ids = store.load_message_ids("s1")
        store.compact_session("s1", summary=_user("[summary]"), compact_ids=all_ids)

        # Find the summary row — it's the one with compacted_by IS NULL
        # and fragment_kind = 'request-message' that wasn't in the originals.
        summary_row = store._con.execute(
            "SELECT id FROM fragments "
            "WHERE session_id = 's1' AND compacted_by IS NULL AND fragment_kind = 'request-message'"
        ).fetchone()
        assert summary_row is not None

        # Count compacted rows before.
        compacted_before = store._con.execute(
            "SELECT COUNT(*) AS cnt FROM fragments WHERE compacted_by IS NOT NULL"
        ).fetchone()
        assert compacted_before is not None
        assert compacted_before["cnt"] > 0

        # Delete the summary row — should cascade to compacted rows.
        store._con.execute("PRAGMA foreign_keys = ON")
        store._con.execute("DELETE FROM fragments WHERE id = ?", [summary_row["id"]])
        store._con.commit()

        # All compacted_by children should be gone.
        compacted_after = store._con.execute(
            "SELECT COUNT(*) AS cnt FROM fragments WHERE compacted_by IS NOT NULL"
        ).fetchone()
        assert compacted_after is not None
        assert compacted_after["cnt"] == 0


def test_nested_compaction() -> None:
    """Compact, add more messages, compact again — both summaries
    coexist and load order is correct.
    """
    with SessionStore() as store:
        # Round 1: 4 messages, compact all.
        msgs1 = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
        store.save_messages("s1", msgs1)
        ids1 = store.load_message_ids("s1")
        store.compact_session("s1", summary=_user("[summary 1]"), compact_ids=ids1)

        # Round 2: add more messages.
        msgs2 = [_user("q3"), _assistant("a3"), _user("q4"), _assistant("a4")]
        store.save_messages("s1", msgs2)

        # Compact again: summary1 + 4 new messages → keep last 2, compact the rest.
        ids2 = store.load_message_ids("s1")
        # Compact summary1 + q3 + a3 (first 3 of current visible messages).
        store.compact_session("s1", summary=_user("[summary 2]"), compact_ids=ids2[:3])

        loaded = store.load_messages("s1")

    # Should have: summary2 + q4 + a4
    assert len(loaded) == 3
    first = loaded[0]
    assert isinstance(first, ModelRequest)
    assert any(
        isinstance(p, UserPromptPart) and isinstance(p.content, str) and "[summary 2]" in p.content
        for p in first.parts
    )


def test_listing_counts_all_messages_including_compacted() -> None:
    """list_sessions message_count includes compacted rows — it shows
    total lifetime messages, not just active ones.
    """
    with SessionStore() as store:
        msgs = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
        store.save_messages("s1", msgs)

        all_ids = store.load_message_ids("s1")
        store.compact_session("s1", summary=_user("[summary]"), compact_ids=all_ids[:2])

        sessions = store.list_sessions()

    assert len(sessions) == 1
    # 4 original + 1 summary = 5 total message-level rows.
    assert sessions[0].message_count == 5


def test_list_sessions_cost_with_compacted_rows() -> None:
    """Session cost aggregation includes compacted response rows."""
    with SessionStore() as store:
        store.save_messages(
            "s1",
            [_user("q1"), _assistant("a1")],
            cost=0.01,
            model_name="m1",
        )
        store.save_messages(
            "s1",
            [_user("q2"), _assistant("a2")],
            cost=0.02,
            model_name="m1",
        )

        # Compact the first turn.
        ids = store.load_message_ids("s1")
        store.compact_session("s1", summary=_user("[summary]"), compact_ids=ids[:2])

        sessions = store.list_sessions()

    assert len(sessions) == 1
    # Both costs should be in the total — compacted rows aren't deleted.
    assert sessions[0].total_cost == pytest.approx(0.03)


# ═══════════════════════════════════════════════════════════════════════
# Streaming: edge cases
# ═══════════════════════════════════════════════════════════════════════


def test_streaming_tool_call_parts() -> None:
    """Streaming a response with tool call parts round-trips correctly."""
    with SessionStore() as store:
        store.set_context("s1")
        store.save_messages("s1", [_user("analyse code")])

        writer = store.begin_response("s1", model_name="test")
        tc = ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1")
        writer.add_part(0, tc)
        writer.finish_part(0, tc)
        text = TextPart(content="")
        writer.add_part(1, text)
        writer.finish_part(1, TextPart(content="Here's the file."))
        writer.finish()

        loaded = store.load_messages("s1")

    response = loaded[1]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 2
    assert isinstance(response.parts[0], ToolCallPart)
    assert response.parts[0].tool_name == "read_file"
    assert isinstance(response.parts[1], TextPart)
    assert response.parts[1].content == "Here's the file."


def test_streaming_finish_idempotent() -> None:
    """Calling finish() multiple times doesn't create duplicates."""
    with SessionStore() as store:
        store.set_context("s1")
        store.save_messages("s1", [_user("hello")])

        writer = store.begin_response("s1", model_name="test")
        writer.add_part(0, TextPart(content=""))
        writer.finish_part(0, TextPart(content="hi"))
        writer.finish()
        writer.finish()
        writer.finish(cost=0.001)

        loaded = store.load_messages("s1")

    # Still just 2 messages (request + response), not duplicated.
    assert len(loaded) == 2


def test_streaming_two_responses_in_sequence() -> None:
    """Two streamed responses from different turns coexist."""
    with SessionStore() as store:
        store.set_context("s1")

        # Turn 1.
        store.save_messages("s1", [_user("q1")])
        w1 = store.begin_response("s1", model_name="test")
        w1.add_part(0, TextPart(content=""))
        w1.finish_part(0, TextPart(content="a1"))
        w1.finish()

        # Turn 2.
        store.save_messages("s1", [_user("q2")])
        w2 = store.begin_response("s1", model_name="test")
        w2.add_part(0, TextPart(content=""))
        w2.finish_part(0, TextPart(content="a2"))
        w2.finish()

        loaded = store.load_messages("s1")

    assert len(loaded) == 4  # q1, a1, q2, a2
    assert isinstance(loaded[1], ModelResponse)
    assert loaded[1].parts[0].content == "a1"  # type: ignore[union-attr]
    assert isinstance(loaded[3], ModelResponse)
    assert loaded[3].parts[0].content == "a2"  # type: ignore[union-attr]
