"""Tests for session serialisation — the pure data-transformation layer.

Data-first: realistic message and incident data defined as constants.
Tests verify Fragment construction and round-trip serialisation
without touching SQLite.

Organisation:
- Message serialisation (prepare_message_row, prepare_part_rows, reconstruct_message)
- Incident round-trips (FailedAttempt, HistoryRepair)
- Input row construction (prepare_input_row)
- Incident row construction (prepare_incident_row)
- Tool-call args validation
"""

from __future__ import annotations

import json

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage
from pytest_cases import parametrize_with_cases

from rbtr_legacy.sessions.incidents import (
    FailedAttempt,
    FailureKind,
    HistoryRepair,
    Incident,
    IncidentOutcome,
    RecoveryStrategy,
)
from rbtr_legacy.sessions.kinds import FragmentKind, FragmentStatus, SessionContext
from rbtr_legacy.sessions.serialise import (
    dump_part,
    prepare_incident_row,
    prepare_input_row,
    prepare_message_row,
    prepare_part_rows,
    reconstruct_message,
)
from tests.engine.builders import _assistant, _user

# ── Shared data ──────────────────────────────────────────────────────

_CTX = SessionContext(
    session_id="s1",
    session_label="test-session",
    repo_owner="acme",
    repo_name="app",
    model_name="claude/sonnet",
    review_target="PR #42",
)

_USAGE = RequestUsage(input_tokens=100, output_tokens=50)


# ═══════════════════════════════════════════════════════════════════════
# Message serialisation
# ═══════════════════════════════════════════════════════════════════════


def test_prepare_message_row_request() -> None:
    """Request message row has correct kind and self-referencing ID."""
    msg = _user("hello")
    row = prepare_message_row(msg, context=_CTX, row_id="r1")

    assert row.id == "r1"
    assert row.message_id == "r1"
    assert row.fragment_index == 0
    assert row.fragment_kind == FragmentKind.REQUEST_MESSAGE
    assert row.session_id == "s1"
    assert row.session_label == "test-session"
    assert row.model_name == "claude/sonnet"
    assert row.status == FragmentStatus.COMPLETE
    assert row.input_tokens is None
    assert row.data_json is not None


def test_prepare_message_row_response_tokens() -> None:
    """Response message row captures token counts from usage."""
    msg = _assistant("hi", usage=_USAGE)
    row = prepare_message_row(msg, context=_CTX, row_id="r2")

    assert row.fragment_kind == FragmentKind.RESPONSE_MESSAGE
    assert row.input_tokens == 100
    assert row.output_tokens == 50


def test_prepare_message_row_failed_status() -> None:
    """Failed status is propagated to the row."""
    msg = _user("fail")
    row = prepare_message_row(msg, context=_CTX, row_id="r3", status=FragmentStatus.FAILED)

    assert row.status == FragmentStatus.FAILED


def test_prepare_part_rows_count() -> None:
    """Part rows match the number of parts, starting at index 1."""
    msg = ModelResponse(
        parts=[
            ThinkingPart(content="hmm"),
            TextPart(content="answer"),
            ToolCallPart(tool_name="grep", args={"q": "x"}, tool_call_id="tc1"),
        ],
        usage=_USAGE,
        model_name="test",
    )
    rows = prepare_part_rows(msg, message_id="m1", context=_CTX)

    assert len(rows) == 3
    assert [r.fragment_index for r in rows] == [1, 2, 3]
    assert rows[0].fragment_kind == FragmentKind.THINKING
    assert rows[1].fragment_kind == FragmentKind.TEXT
    assert rows[2].fragment_kind == FragmentKind.TOOL_CALL
    assert rows[2].tool_name == "grep"


def test_prepare_part_rows_user_text() -> None:
    """UserPromptPart populates the `user_text` column."""
    msg = _user("search for bugs")
    rows = prepare_part_rows(msg, message_id="m1", context=_CTX)

    assert len(rows) == 1
    assert rows[0].user_text == "search for bugs"


def test_prepare_part_rows_tool_name_on_return_and_retry() -> None:
    """ToolReturnPart and RetryPromptPart set `tool_name`."""
    msg = ModelRequest(
        parts=[
            ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="tc1"),
            RetryPromptPart(content="bad", tool_name="grep", tool_call_id="tc2"),
        ]
    )
    rows = prepare_part_rows(msg, message_id="m1", context=_CTX)

    assert rows[0].tool_name == "read_file"
    assert rows[1].tool_name == "grep"


def test_reconstruct_message_roundtrip() -> None:
    """Header + parts round-trip through reconstruct_message."""
    msg = ModelResponse(
        parts=[TextPart(content="hello"), ThinkingPart(content="deep thought")],
        usage=_USAGE,
        model_name="test",
    )
    row = prepare_message_row(msg, context=_CTX, row_id="r1")
    part_rows = prepare_part_rows(msg, message_id="r1", context=_CTX)
    assert row.data_json is not None

    restored = reconstruct_message(
        row.fragment_kind,
        row.data_json,
        [r.data_json for r in part_rows if r.data_json],
    )

    assert restored == msg


@parametrize_with_cases("history", cases="tests.sessions.case_histories")
def test_serialise_roundtrip_all_history_shapes(history: list[ModelMessage]) -> None:
    """Every message in every history shape survives serialise → reconstruct."""
    for msg in history:
        kind = (
            FragmentKind.REQUEST_MESSAGE
            if isinstance(msg, ModelRequest)
            else FragmentKind.RESPONSE_MESSAGE
        )
        row = prepare_message_row(msg, context=_CTX, row_id="r")
        part_rows = prepare_part_rows(msg, message_id="r", context=_CTX)
        assert row.data_json is not None

        restored = reconstruct_message(
            kind,
            row.data_json,
            [r.data_json for r in part_rows if r.data_json],
        )
        assert restored == msg


def test_dump_part_json_valid() -> None:
    """dump_part produces valid JSON for each part type."""
    parts: list[ModelRequestPart | ModelResponsePart] = [
        UserPromptPart(content="hi"),
        TextPart(content="hello"),
        ToolCallPart(tool_name="f", args={"x": 1}, tool_call_id="tc1"),
    ]
    for part in parts:
        result = dump_part(part)
        parsed = json.loads(result)
        assert "part_kind" in parsed


# ═══════════════════════════════════════════════════════════════════════
# Incident round-trips
# ═══════════════════════════════════════════════════════════════════════

# ── FailedAttempt ─────────────────────────────────────────────

_ATTEMPT_FAILED = FailedAttempt(
    turn_id="t1",
    failure_kind=FailureKind.OVERFLOW,
    strategy=RecoveryStrategy.COMPACT_THEN_RETRY,
    model_name="claude/sonnet",
    status_code=400,
    error_text="context too long",
    diagnostic="Traceback ...",
    history_message_count=12,
    outcome=IncidentOutcome.RECOVERED,
)

_ATTEMPT_FAILED_MINIMAL = FailedAttempt(
    turn_id="t2",
    failure_kind=FailureKind.TOOL_ARGS,
    strategy=RecoveryStrategy.SIMPLIFY_HISTORY,
)


def test_attempt_failed_roundtrip() -> None:
    """Full FailedAttempt serialises and deserialises losslessly."""
    j = _ATTEMPT_FAILED.model_dump_json(exclude_none=True)
    restored = FailedAttempt.model_validate_json(j)

    assert restored.turn_id == "t1"
    assert restored.failure_kind is FailureKind.OVERFLOW
    assert restored.strategy is RecoveryStrategy.COMPACT_THEN_RETRY
    assert restored.outcome is IncidentOutcome.RECOVERED
    assert restored.status_code == 400
    assert restored.model_name == "claude/sonnet"


def test_attempt_failed_minimal_excludes_none() -> None:
    """Minimal payload omits None fields in JSON."""
    j = _ATTEMPT_FAILED_MINIMAL.model_dump_json(exclude_none=True)
    data = json.loads(j)

    assert set(data.keys()) == {"turn_id", "failure_kind", "strategy"}
    assert "model_name" not in data
    assert "status_code" not in data


def test_attempt_failed_ignores_extra_fields() -> None:
    """Unknown fields are ignored on deserialisation (forward compat)."""
    j = '{"turn_id": "t1", "failure_kind": "overflow", "strategy": "none", "future_field": 42}'
    restored = FailedAttempt.model_validate_json(j)

    assert restored.turn_id == "t1"
    assert not hasattr(restored, "future_field")


# ── HistoryRepair ─────────────────────────────────────────────

_REPAIR_DANGLING = HistoryRepair(
    strategy=RecoveryStrategy.REPAIR_DANGLING,
    tool_names=["read_file", "grep"],
    call_count=2,
    reason="cancelled_mid_tool_call",
)

_REPAIR_DEMOTE = HistoryRepair(
    strategy=RecoveryStrategy.DEMOTE_THINKING,
    parts_demoted=5,
    reason="cross_provider_retry",
)

_REPAIR_FLATTEN = HistoryRepair(
    strategy=RecoveryStrategy.FLATTEN_TOOL_EXCHANGES,
    tool_calls_flattened=3,
    tool_returns_flattened=3,
    retry_prompts_dropped=1,
    reason="cross_provider_retry",
)


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("dangling", _REPAIR_DANGLING),
        ("demote", _REPAIR_DEMOTE),
        ("flatten", _REPAIR_FLATTEN),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_history_repair_roundtrip(label: str, payload: HistoryRepair) -> None:
    """Each repair strategy round-trips through JSON."""
    j = payload.model_dump_json(exclude_none=True)
    restored = HistoryRepair.model_validate_json(j)

    assert restored.strategy == payload.strategy
    assert restored.reason == payload.reason


def test_repair_dangling_fields() -> None:
    """REPAIR_DANGLING payload preserves tool names and count."""
    j = _REPAIR_DANGLING.model_dump_json(exclude_none=True)
    restored = HistoryRepair.model_validate_json(j)

    assert restored.tool_names == ["read_file", "grep"]
    assert restored.call_count == 2


def test_repair_flatten_fields() -> None:
    """FLATTEN_TOOL_EXCHANGES payload preserves all counters."""
    j = _REPAIR_FLATTEN.model_dump_json(exclude_none=True)
    restored = HistoryRepair.model_validate_json(j)

    assert restored.tool_calls_flattened == 3
    assert restored.tool_returns_flattened == 3
    assert restored.retry_prompts_dropped == 1


def test_repair_ignores_extra_fields() -> None:
    """Unknown fields are ignored on deserialisation (forward compat)."""
    j = '{"strategy": "repair_dangling", "new_field": true}'
    restored = HistoryRepair.model_validate_json(j)

    assert restored.strategy is RecoveryStrategy.REPAIR_DANGLING


# ═══════════════════════════════════════════════════════════════════════
# Input row construction
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "kind",
    [FragmentKind.COMMAND, FragmentKind.SHELL],
)
def test_prepare_input_row(kind: FragmentKind) -> None:
    """Input rows have correct kind, user_text, and no data_json."""
    row = prepare_input_row("/review 42", kind, context=_CTX, row_id="i1")

    assert row.id == "i1"
    assert row.message_id == "i1"
    assert row.fragment_index == 0
    assert row.fragment_kind == kind
    assert row.user_text == "/review 42"
    assert row.data_json is None
    assert row.status == FragmentStatus.COMPLETE
    assert row.session_id == "s1"


# ═══════════════════════════════════════════════════════════════════════
# Incident row construction
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("label", "kind", "payload"),
    [
        (
            "attempt_failed",
            FragmentKind.LLM_ATTEMPT_FAILED,
            _ATTEMPT_FAILED,
        ),
        (
            "history_repair",
            FragmentKind.LLM_HISTORY_REPAIR,
            _REPAIR_DANGLING,
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_prepare_incident_row(label: str, kind: FragmentKind, payload: Incident) -> None:
    """Incident rows are self-referencing with serialised payload."""
    row = prepare_incident_row(kind, payload, context=_CTX, row_id="a1")

    assert row.id == "a1"
    assert row.message_id == "a1"
    assert row.fragment_index == 0
    assert row.fragment_kind == kind
    assert row.status == FragmentStatus.COMPLETE
    assert row.data_json is not None
    assert row.user_text is None

    # data_json is valid JSON matching the payload.
    parsed = json.loads(row.data_json)
    assert "strategy" in parsed


def test_incident_row_excludes_none() -> None:
    """Incident row data_json omits None fields."""
    row = prepare_incident_row(
        FragmentKind.LLM_ATTEMPT_FAILED,
        _ATTEMPT_FAILED_MINIMAL,
        context=_CTX,
        row_id="a2",
    )
    assert row.data_json is not None
    parsed = json.loads(row.data_json)
    assert "model_name" not in parsed
    assert "status_code" not in parsed


def test_incident_row_payload_roundtrip() -> None:
    """Payload stored in incident row can be deserialised back."""
    row = prepare_incident_row(
        FragmentKind.LLM_ATTEMPT_FAILED,
        _ATTEMPT_FAILED,
        context=_CTX,
        row_id="a3",
    )
    assert row.data_json is not None
    restored = FailedAttempt.model_validate_json(row.data_json)

    assert restored.failure_kind is FailureKind.OVERFLOW
    assert restored.outcome is IncidentOutcome.RECOVERED


# ═══════════════════════════════════════════════════════════════════════
# Tool-call args validation
# ═══════════════════════════════════════════════════════════════════════


def test_corrupt_tool_args_preserved_on_deserialise() -> None:
    """reconstruct_message preserves corrupt args for upstream repair.

    Args validation is not a deserialisation concern — it runs in
    `_prepare_turn` via `validate_tool_call_args` where the
    repair can be recorded as an incident.
    """
    corrupt_part = ToolCallPart(
        tool_name="read_file",
        args='{"path": bad json',
        tool_call_id="tc1",
    )
    good_response = ModelResponse(
        parts=[corrupt_part],
        usage=_USAGE,
        model_name="test",
    )
    row = prepare_message_row(good_response, context=_CTX, row_id="r1")
    part_rows = prepare_part_rows(good_response, message_id="r1", context=_CTX)
    assert row.data_json is not None

    restored = reconstruct_message(
        row.fragment_kind,
        row.data_json,
        [r.data_json for r in part_rows if r.data_json],
    )
    assert isinstance(restored, ModelResponse)
    # Corrupt args survive deserialisation — repaired later in _prepare_turn.
    assert restored.parts[0].args == '{"path": bad json'  # type: ignore[union-attr]
