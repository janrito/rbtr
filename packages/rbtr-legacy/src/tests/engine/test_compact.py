"""Tests for context compaction — history helpers and engine integration.

Pure functions are tested with realistic message data.  Integration
tests go through `compact_history` and check emitted events + state,
mocking only the LLM call boundary (`_stream_summary`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pytest_cases import parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr_legacy.config import config
from rbtr_legacy.engine.core import Engine
from rbtr_legacy.engine.types import TaskType
from rbtr_legacy.events import (
    CompactionFinished,
    CompactionStarted,
    Output,
    OutputLevel,
    TaskFinished,
)
from rbtr_legacy.llm.compact import compact_agent, compact_history, find_fit_count, reset_compaction
from rbtr_legacy.llm.context import LLMContext
from rbtr_legacy.sessions.history import (
    build_summary_message,
    estimate_tokens,
    serialise_for_summary,
    snap_to_safe_boundary,
    split_history,
    strip_orphaned_tool_returns,
)
from rbtr_legacy.sessions.kinds import SUMMARY_MARKER
from tests.engine.builders import (
    _USAGE,
    _assistant,
    _seed,
    _tool_call_only,
    _tool_result,
    _tool_then_text_model,
    _turns,
    _user,
)
from tests.helpers import StubProvider, drain, has_event_type, output_texts
from tests.sessions.assertions import assert_ordering
from tests.sessions.case_histories import case_single_tool, case_tool_boundary_straddle


@pytest.fixture(autouse=True)
def _disable_memory() -> None:
    """Compact tests focus on compaction logic, not fact extraction."""
    config.memory.enabled = False


# Test data is centralised in tests/sessions/case_histories.py.
# Specific histories are imported as case functions and called
# where needed.


# ═══════════════════════════════════════════════════════════════════════
# split_history
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("total_turns", "keep", "expected_old_turns"),
    [
        (1, 5, 0),  # fewer than keep → nothing to compact
        (5, 5, 0),  # exactly keep → nothing to compact
        (6, 3, 3),  # standard split
        (10, 2, 8),  # keep very few
        (5, 1, 4),  # keep only last turn
        (2, 1, 1),  # minimal compactable history
    ],
)
def test_split_history_parametrized(total_turns: int, keep: int, expected_old_turns: int) -> None:
    history = _turns(total_turns)
    old, kept = split_history(history, keep_turns=keep)
    assert len(old) == expected_old_turns * 2
    assert len(kept) == (total_turns - expected_old_turns) * 2
    assert old + kept == history


def test_split_history_empty() -> None:
    old, kept = split_history([], keep_turns=5)
    assert old == []
    assert kept == []


def test_split_history_tool_requests_not_turn_boundaries() -> None:
    """Tool-return-only ModelRequests don't start new turns."""
    history: list[ModelMessage] = [
        _user("q1"),
        _tool_call_only("read_file", {"path": "a.py"}),
        _tool_result("read_file", "content"),
        _assistant("a1"),
        _user("q2"),
        _assistant("a2"),
    ]
    old, kept = split_history(history, keep_turns=1)
    assert old == history[:4]  # first turn with its tool calls
    assert kept == history[4:]  # second turn only


@parametrize_with_cases("history", cases=[case_single_tool])
def test_split_history_realistic(history: list[ModelMessage]) -> None:
    """Split the realistic conversation — keep 1 turn, compact the rest."""
    old, kept = split_history(history, keep_turns=1)
    # Last user turn starts at "Any issues?"
    last_user_idx = next(
        i
        for i in reversed(range(len(history)))
        if isinstance(history[i], ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in history[i].parts)
    )
    assert kept == history[last_user_idx:]
    assert old == history[:last_user_idx]


@parametrize_with_cases("history", cases="tests.sessions.case_histories")
def test_split_history_all_shapes(history: list[ModelMessage]) -> None:
    """split_history produces valid old + kept for every history shape."""
    old, kept = split_history(history, keep_turns=1)

    # old + kept = original (no messages lost or duplicated).
    assert old + kept == history

    # kept is structurally valid if non-empty.
    if kept:
        assert_ordering(kept)

    # old is structurally valid if non-empty.
    if old:
        assert_ordering(old)


# ── Orphaned tool return tests ───────────────────────────────────────
#
# After compaction, a ToolReturnPart can reference a ToolCallPart that
# was compacted away (in a preceding ModelResponse).  The API rejects
# this: "No tool call found for function call output with call_id …".
#
# split_history must move such orphaned tool-return requests into old.
# These tests build realistic multi-turn conversations with explicit
# tool_call_ids and verify no orphans survive in kept.


def _assert_no_orphaned_returns(kept: list[ModelMessage]) -> None:
    """Assert every ToolReturnPart in kept has a matching ToolCallPart."""
    call_ids: set[str] = set()
    for msg in kept:
        if isinstance(msg, ModelResponse):
            for resp_part in msg.parts:
                if isinstance(resp_part, ToolCallPart) and resp_part.tool_call_id:
                    call_ids.add(resp_part.tool_call_id)
    for msg in kept:
        if isinstance(msg, ModelRequest):
            for req_part in msg.parts:
                if isinstance(req_part, ToolReturnPart) and req_part.tool_call_id:
                    assert req_part.tool_call_id in call_ids, (
                        f"Orphaned tool return: {req_part.tool_call_id}"
                    )


def test_orphan_single_tool_return_at_boundary() -> None:
    """A lone tool return right after the cut point is moved to old.

    Turn 1 ends with a tool call; its return sits just before turn 2.
    Keeping only turn 2 must move the orphaned return into old.
    """
    history: list[ModelMessage] = [
        _user("turn 1"),  # [0]
        _tool_call_only("grep", call_id="call_A"),  # [1]
        _tool_result("grep", "result", call_id="call_A"),  # [2] — orphan candidate
        _assistant("answer 1"),  # [3]
        _user("turn 2"),  # [4]
        _assistant("answer 2"),  # [5]
    ]
    old, kept = split_history(history, keep_turns=1)
    assert kept == [history[4], history[5]]
    assert history[2] in old  # orphan moved to old
    _assert_no_orphaned_returns(kept)


def test_orphan_multiple_consecutive_returns() -> None:
    """Multiple orphaned tool returns in a row are all moved to old.

    Model made two parallel tool calls; both returns are orphaned
    when the response containing the calls is compacted.
    """
    history: list[ModelMessage] = [
        _user("turn 1"),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="grep", args={}, tool_call_id="call_A"),
                ToolCallPart(tool_name="read", args={}, tool_call_id="call_B"),
            ],
            usage=_USAGE,
            model_name="test",
        ),
        _tool_result("grep", "result", call_id="call_A"),
        _tool_result("read", "result", call_id="call_B"),
        _assistant("done"),
        _user("turn 2"),
        _assistant("answer 2"),
    ]
    old, kept = split_history(history, keep_turns=1)
    assert kept == history[-2:]
    assert history[2] in old
    assert history[3] in old
    _assert_no_orphaned_returns(kept)


def test_orphan_not_moved_when_call_in_kept() -> None:
    """A tool return whose call is also in kept stays in kept.

    Both the ToolCallPart and ToolReturnPart are within the same
    kept turn — no orphan.
    """
    history: list[ModelMessage] = [
        _user("turn 1"),
        _assistant("answer 1"),
        _user("turn 2"),
        _tool_call_only("grep", call_id="call_A"),
        _tool_result("grep", "result", call_id="call_A"),
        _assistant("answer 2"),
    ]
    old, kept = split_history(history, keep_turns=1)
    assert old == history[:2]
    assert kept == history[2:]
    _assert_no_orphaned_returns(kept)


def test_orphan_mixed_orphaned_and_paired() -> None:
    """Only orphaned returns move; paired ones stay.

    Turn 1 makes call_A.  Turn 2 makes call_B.  Keep turn 2.
    call_A's return (between turns) is orphaned; call_B's is not.
    """
    history: list[ModelMessage] = [
        _user("turn 1"),
        _tool_call_only("grep", call_id="call_A"),
        _tool_result("grep", "result", call_id="call_A"),
        _assistant("answer 1"),
        _user("turn 2"),
        _tool_call_only("read", call_id="call_B"),
        _tool_result("read", "result", call_id="call_B"),
        _assistant("answer 2"),
    ]
    old, kept = split_history(history, keep_turns=1)
    # call_A's return should be in old, call_B's in kept.
    assert history[2] in old  # call_A return → orphaned
    assert history[6] in kept  # call_B return → paired
    _assert_no_orphaned_returns(kept)


def test_orphan_chain_across_turns() -> None:
    """Orphan cleanup across a three-turn history with keep_turns=1.

    Turn 1: call_A → return_A
    Turn 2: call_B → return_B
    Turn 3: text only

    Keep 1 → turns 1 and 2 compacted.  return_B sits between
    turn 2's response and turn 3's user prompt.
    """
    history: list[ModelMessage] = [
        _user("turn 1"),
        _tool_call_only("grep", call_id="call_A"),
        _tool_result("grep", "result", call_id="call_A"),
        _assistant("answer 1"),
        _user("turn 2"),
        _tool_call_only("read", call_id="call_B"),
        _tool_result("read", "result", call_id="call_B"),
        _assistant("answer 2"),
        _user("turn 3"),
        _assistant("answer 3"),
    ]
    _old, kept = split_history(history, keep_turns=1)
    assert kept == history[-2:]
    _assert_no_orphaned_returns(kept)


def test_orphan_with_response_between_return_and_next_turn() -> None:
    """Orphaned return followed by a response then the next turn.

    The return is orphaned but the response after it is not a
    tool-return request — the while-loop should stop at the response.
    """
    history: list[ModelMessage] = [
        _user("turn 1"),
        _tool_call_only("grep", call_id="call_A"),
        _tool_result("grep", "result", call_id="call_A"),
        _assistant("continuation"),  # response after tool return, same turn
        _user("turn 2"),
        _assistant("answer 2"),
    ]
    # Keep 1 — turn 1 (including all its messages) goes to old.
    _old, kept = split_history(history, keep_turns=1)
    assert kept == history[-2:]
    _assert_no_orphaned_returns(kept)


def test_orphan_tool_return_with_none_call_id_not_moved() -> None:
    """A ToolReturnPart with tool_call_id=None is not treated as orphaned.

    Some older pydantic-ai versions or custom agents may produce
    tool returns without call IDs.  These should stay in kept.
    """
    history: list[ModelMessage] = [
        _user("turn 1"),
        _assistant("answer 1"),
        _user("turn 2"),
        # Tool return without a call_id — not orphaned by our definition.
        ModelRequest(parts=[ToolReturnPart(tool_name="grep", content="x", tool_call_id=None)]),  # type: ignore[arg-type]  # testing None edge case
        _assistant("answer 2"),
    ]
    _old, kept = split_history(history, keep_turns=1)
    # The None-id return stays in kept (no matching logic applies).
    assert any(
        isinstance(p, ToolReturnPart) and p.tool_call_id is None
        for msg in kept
        if isinstance(msg, ModelRequest)
        for p in msg.parts
    )


def test_orphan_user_prompt_stops_migration() -> None:
    """The while-loop stops at a UserPromptPart — never moves user turns.

    Even if a tool return appears AFTER a user prompt (unusual but
    possible), the user prompt is never moved to old.
    """
    history: list[ModelMessage] = [
        _user("turn 1"),
        _tool_call_only("grep", call_id="call_A"),
        _tool_result("grep", "result", call_id="call_A"),
        _assistant("answer 1"),
        _user("turn 2"),
        _assistant("answer 2"),
        _user("turn 3"),
        _assistant("answer 3"),
    ]
    _old, kept = split_history(history, keep_turns=2)
    # Kept must start at "turn 2" — never moves a UserPromptPart.
    assert isinstance(kept[0], ModelRequest)
    assert any(isinstance(p, UserPromptPart) for p in kept[0].parts)
    _assert_no_orphaned_returns(kept)


def test_orphan_multi_step_tool_chain_within_turn() -> None:
    """Multi-step tool chain within one turn — all returns stay paired.

    Model calls tool A, gets result, calls tool B, gets result, answers.
    All within one turn.  Nothing is orphaned.
    """
    history: list[ModelMessage] = [
        _user("turn 1"),
        _assistant("simple"),
        _user("turn 2"),
        _tool_call_only("grep", call_id="call_A"),
        _tool_result("grep", "result", call_id="call_A"),
        _tool_call_only("read", call_id="call_B"),
        _tool_result("read", "result", call_id="call_B"),
        _assistant("done"),
    ]
    _old, kept = split_history(history, keep_turns=1)
    assert kept == history[2:]
    _assert_no_orphaned_returns(kept)


def test_orphan_all_messages_kept_no_change() -> None:
    """When nothing is compacted, no orphan cleanup is needed."""
    history: list[ModelMessage] = [
        _user("turn 1"),
        _tool_call_only("grep", call_id="call_A"),
        _tool_result("grep", "result", call_id="call_A"),
        _assistant("answer"),
    ]
    old, kept = split_history(history, keep_turns=5)
    assert old == []
    assert kept == history


def test_orphan_reproduces_real_bug() -> None:
    """Reproduces the exact pattern from the production bug.

    Turn N (compacted):
        ... earlier messages ...
        ModelResponse with ToolCallPart(list_files, call_xJJ)

    Turn N+1 (kept):
        ModelRequest with UserPromptPart("check notes")
        ModelResponse with TextPart("no notes found")
        ModelRequest with ToolReturnPart(list_files, call_xJJ)  ← orphan!

    The orphaned return must be moved to old so the API never sees
    a tool return without its matching tool call.
    """
    history: list[ModelMessage] = [
        # Earlier turns — will be compacted.
        _user("review the PR"),
        _tool_call_only("diff", call_id="call_111"),
        _tool_result("diff", "result", call_id="call_111"),
        _assistant("Here's the diff."),
        _tool_call_only("list_files", call_id="call_xJJ"),
        # ↑ This response has the tool call for list_files.
        # ↓ The user prompt starts a new turn, splitting here.
        _user("check notes"),
        _assistant("No notes found."),
        # The tool return for call_xJJ arrives AFTER the user prompt
        # (pydantic-ai structures it as a separate ModelRequest).
        _tool_result("list_files", "result", call_id="call_xJJ"),
        # Another turn.
        _user("any security concerns?"),
        _assistant("No issues."),
    ]
    _old, kept = split_history(history, keep_turns=1)
    # The orphan (call_xJJ return) must not be in kept.
    _assert_no_orphaned_returns(kept)
    assert kept == history[-2:]

    # With keep_turns=2, keep "check notes" and "security" turns.
    # The orphaned return for call_xJJ should still be in old.
    _old2, kept2 = split_history(history, keep_turns=2)
    _assert_no_orphaned_returns(kept2)
    # kept starts at "check notes"
    assert isinstance(kept2[0], ModelRequest)
    assert any(isinstance(p, UserPromptPart) for p in kept2[0].parts)


# ═══════════════════════════════════════════════════════════════════════
# serialise_for_summary
# ═══════════════════════════════════════════════════════════════════════


@parametrize_with_cases("history", cases=[case_single_tool])
def test_serialise_realistic_conversation(history: list[ModelMessage]) -> None:
    """Serialise the full realistic history and check all section types."""
    result = serialise_for_summary(history)
    assert "## User\nRead foo.py" in result
    assert "## Assistant\nLet me read that." in result
    assert '## Tool call: read_file({"path": "foo.py"})' in result
    assert "## Tool result: read_file\ndef main(): pass" in result


def test_serialise_truncates_tool_results() -> None:
    """Tool results exceeding max_tool_chars are truncated."""
    long_diff = "x" * 5_000
    messages: list[ModelMessage] = [_tool_result("diff", long_diff)]
    result = serialise_for_summary(messages, max_tool_chars=100)
    assert "…[truncated]" in result
    # The truncated result should be ~100 chars of content + marker
    tool_section = result.split("\n", 1)[1]  # skip "## Tool result: diff"
    assert len(tool_section) < 200


def test_serialise_empty() -> None:
    assert serialise_for_summary([]) == ""


def test_serialise_tool_call_none_args() -> None:
    """Tool call with None args produces empty parens."""
    msg = ModelResponse(
        parts=[ToolCallPart(tool_name="list_files", args=None)],
        usage=_USAGE,
        model_name="test",
    )
    result = serialise_for_summary([msg])
    assert "## Tool call: list_files()" in result


# ═══════════════════════════════════════════════════════════════════════
# estimate_tokens
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("abcd", 1),
        ("a" * 100, 25),
        ("a" * 1000, 250),
        ("hello world", 2),  # 11 chars → 2 tokens
    ],
)
def test_estimate_tokens(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


# ═══════════════════════════════════════════════════════════════════════
# build_summary_message
# ═══════════════════════════════════════════════════════════════════════


def test_build_summary_message() -> None:
    msg = build_summary_message("Files discussed: foo.py, bar.py")
    assert isinstance(msg, ModelRequest)
    assert len(msg.parts) == 1
    part = msg.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, str)
    assert SUMMARY_MARKER in part.content
    assert "Files discussed: foo.py, bar.py" in part.content


# ═══════════════════════════════════════════════════════════════════════
# find_fit_count
# ═══════════════════════════════════════════════════════════════════════


def test_find_fit_count_all_fit() -> None:
    """When all messages fit, returns the full count."""
    messages = _turns(3)  # 6 messages, small text
    # 10k tokens is way more than enough
    assert find_fit_count(messages, available_tokens=10_000, max_tool_chars=2_000) == 6


def test_find_fit_count_none_fit() -> None:
    """When even 1 message exceeds budget, returns 0."""
    messages: list[ModelMessage] = [_user("x" * 10_000)]  # ~2500 tokens
    assert find_fit_count(messages, available_tokens=100, max_tool_chars=2_000) == 0


def test_find_fit_count_partial() -> None:
    """Returns the largest prefix that fits."""
    # Each user message is ~250 tokens (1000 chars)
    messages: list[ModelMessage] = [_user("x" * 1000) for _ in range(10)]
    # Budget for ~3 messages worth (header + content ≈ 260 tokens each)
    serialised_3 = serialise_for_summary(messages[:3], max_tool_chars=2_000)
    budget = estimate_tokens(serialised_3)
    result = find_fit_count(messages, available_tokens=budget, max_tool_chars=2_000)
    assert result == 3


def test_find_fit_count_with_tool_results() -> None:
    """Tool result truncation affects what fits."""
    messages: list[ModelRequest | ModelResponse] = [
        _tool_result("diff", "x" * 8_000),
        _tool_result("read_file", "y" * 8_000),
        _tool_result("search", "z" * 8_000),
    ]
    # With max_tool_chars=100, each result is small
    count_small = find_fit_count(messages, available_tokens=500, max_tool_chars=100)
    # With max_tool_chars=8000, each result is huge
    count_large = find_fit_count(messages, available_tokens=500, max_tool_chars=8_000)
    assert count_small > count_large


# ═══════════════════════════════════════════════════════════════════════
# snap_to_safe_boundary
# ═══════════════════════════════════════════════════════════════════════


def test_snap_noop_when_boundary_is_safe() -> None:
    """Splitting after a user prompt or text response is always safe."""
    messages: list[ModelMessage] = [
        _user("hello"),
        _assistant("hi"),
        _user("question"),
        _assistant("answer"),
    ]
    assert snap_to_safe_boundary(messages, 2) == 2
    assert snap_to_safe_boundary(messages, 4) == 4


def test_snap_backs_up_from_tool_call_response() -> None:
    """Splitting right after a tool-call response would orphan its results."""
    messages: list[ModelMessage] = [
        _user("do stuff"),
        _tool_call_only("read_file", {"path": "a.py"}, call_id="tc1"),
        _tool_result("read_file", "contents", call_id="tc1"),
        _assistant("done"),
    ]
    # count=2 → split after _tool_call → unsafe
    assert snap_to_safe_boundary(messages, 2) == 1
    # count=3 → split after _tool_return → safe
    assert snap_to_safe_boundary(messages, 3) == 3


def test_snap_backs_up_through_consecutive_tool_calls() -> None:
    """Multiple consecutive tool-call responses before any results."""
    messages: list[ModelMessage] = [
        _user("q1"),
        _assistant("a1"),
        _tool_call_only("grep", call_id="tc1"),
        _tool_result("grep", "result", call_id="tc1"),
        _tool_call_only("diff", call_id="tc2"),
        _tool_result("diff", "result", call_id="tc2"),
        _assistant("summary"),
    ]
    # count=3 → after first tool_call → back up to 2
    assert snap_to_safe_boundary(messages, 3) == 2
    # count=5 → after second tool_call → back up to 4
    assert snap_to_safe_boundary(messages, 5) == 4
    # count=4 → after first tool_return → safe
    assert snap_to_safe_boundary(messages, 4) == 4


def test_snap_returns_zero_when_all_unsafe() -> None:
    """When the only message is a tool-call response, returns 0."""
    messages: list[ModelMessage] = [
        _tool_call_only("read_file", call_id="tc1"),
        _tool_result("read_file", "contents", call_id="tc1"),
    ]
    assert snap_to_safe_boundary(messages, 1) == 0


def test_snap_zero_stays_zero() -> None:
    """count=0 is always returned as-is."""
    messages: list[ModelMessage] = [_tool_call_only("read_file", call_id="tc1")]
    assert snap_to_safe_boundary(messages, 0) == 0


def test_snap_text_response_is_safe() -> None:
    """A ModelResponse with only text (no tool calls) is a safe boundary."""
    messages: list[ModelMessage] = [
        _user("hello"),
        _assistant("thinking..."),
        _tool_call_only("grep", call_id="tc1"),
        _tool_result("grep", "found", call_id="tc1"),
    ]
    # count=2 → after _assistant (text only) → safe
    assert snap_to_safe_boundary(messages, 2) == 2


def test_snap_realistic_mid_turn() -> None:
    """Realistic mid-turn pattern: multiple tool rounds in one turn."""
    messages: list[ModelMessage] = [
        _user("review PR"),  # 0
        _tool_call_only("diff", call_id="tc1"),  # 1
        _tool_result("diff", "...", call_id="tc1"),  # 2
        _tool_call_only("read_file", call_id="tc2"),  # 3
        _tool_result("read_file", "...", call_id="tc2"),  # 4
        _tool_call_only("grep", call_id="tc3"),  # 5
        _tool_result("grep", "...", call_id="tc3"),  # 6
        _assistant("Here is the review."),  # 7
    ]
    # Safe boundaries: 0, 1, 3, 5, 7, 8
    assert snap_to_safe_boundary(messages, 1) == 1  # after _user → safe
    assert snap_to_safe_boundary(messages, 2) == 1  # after _tool_call → back to 1
    assert snap_to_safe_boundary(messages, 3) == 3  # after _tool_return → safe
    assert snap_to_safe_boundary(messages, 4) == 3  # after _tool_call → back to 3
    assert snap_to_safe_boundary(messages, 5) == 5  # after _tool_return → safe
    assert snap_to_safe_boundary(messages, 6) == 5  # after _tool_call → back to 5
    assert snap_to_safe_boundary(messages, 7) == 7  # after _tool_return → safe
    assert snap_to_safe_boundary(messages, 8) == 8  # after _assistant → safe


# ═══════════════════════════════════════════════════════════════════════
# compact_history — integration via event contract
# ═══════════════════════════════════════════════════════════════════════


def test_compact_no_llm(config_path: Path, engine: Engine, llm_ctx: LLMContext) -> None:
    """Warns when no LLM is connected."""

    compact_history(llm_ctx)
    texts = output_texts(drain(engine.events))
    assert any("No LLM connected" in t for t in texts)


def test_compact_single_turn(config_path: Path, engine: Engine, llm_ctx: LLMContext) -> None:
    """Single-turn history has nothing to compact."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    _seed(engine, _turns(1))

    compact_history(llm_ctx)
    texts = output_texts(drain(engine.events))
    assert any("Nothing to compact" in t for t in texts)


def test_compact_fewer_turns_than_keep_falls_back(
    config_path: Path, engine: Engine, llm_ctx: LLMContext, stub_provider: StubProvider
) -> None:
    """With 2 turns (= keep_turns=2), normal split finds nothing to
    compact, so it falls back to keeping 1 turn.
    """
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    _seed(engine, _turns(2))
    engine.state.usage.context_window = 200_000

    with compact_agent.override(model=TestModel(custom_output_text="Summary of turn 1.")):
        compact_history(llm_ctx)

    all_events = drain(engine.events)

    started = [e for e in all_events if isinstance(e, CompactionStarted)]
    assert len(started) == 1
    assert started[0].old_messages == 2  # 1 turn compacted
    assert started[0].kept_messages == 2  # 1 turn kept


@parametrize_with_cases("history", cases=[case_single_tool])
def test_compact_replaces_history(
    history: list[ModelMessage],
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """After compaction, history = [summary_msg] + kept turns."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    _seed(engine, list(history))
    engine.state.usage.context_window = 200_000

    summary_text = "Reviewed PR #42. Found unused import in foo.py."
    with compact_agent.override(model=TestModel(custom_output_text=summary_text)):
        compact_history(llm_ctx)

    # Load from DB — the source of truth.
    loaded = engine.store.load_messages(engine.state.session_id)

    # First message is the summary
    first = loaded[0]
    assert isinstance(first, ModelRequest)
    part = first.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, str)
    assert SUMMARY_MARKER in part.content
    assert "Reviewed PR #42" in part.content

    # History is shorter than before, structurally valid.
    assert len(loaded) < len(history)
    assert_ordering(loaded)

    # Last message is preserved (the last assistant response)
    last = loaded[-1]
    assert isinstance(last, ModelResponse)
    assert any(isinstance(p, TextPart) and "No issues found" in p.content for p in last.parts)


def test_compact_emits_both_events(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """Both CompactionStarted and CompactionFinished are emitted."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    _seed(engine, _turns(15))
    engine.state.usage.context_window = 200_000

    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(llm_ctx)
    all_events = drain(engine.events)
    assert has_event_type(all_events, CompactionStarted)
    assert has_event_type(all_events, CompactionFinished)

    finished = [e for e in all_events if isinstance(e, CompactionFinished)]
    assert finished[0].summary_tokens > 0


def test_compact_extra_instructions_in_prompt(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """Extra instructions appear in the prompt sent to the model."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    _seed(engine, _turns(15))
    engine.state.usage.context_window = 200_000

    captured_instructions: list[str] = []

    async def _capture_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        if info.instructions:
            captured_instructions.append(info.instructions)
        yield "Summary."

    stub_provider.set_model(FunctionModel(stream_function=_capture_stream))
    compact_history(llm_ctx, "Focus on security")

    assert any("Focus on security" in c for c in captured_instructions)


def test_compact_over_limit_shrinks_old(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """When serialised old exceeds context, only a fitting prefix is summarised."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    # 10 turns with large user messages (each ~2500 tokens after 4-char heuristic)
    big_history: list[ModelRequest | ModelResponse] = []
    for i in range(10):
        big_history.append(_user(f"{'x' * 10_000} question {i}"))
        big_history.append(_assistant(f"answer {i}"))
    _seed(engine, big_history)
    # Context window that can't fit all old messages after reserve.
    # 10 turns, keep 1 → 9 turns old. Each turn serialises to ~2500 tokens.
    # 9 turns ≈ 22.5k tokens. Set available to ~7.5k so only ~3 turns fit.
    engine.state.usage.context_window = 23_500  # minus 16k reserve = 7.5k available

    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(llm_ctx)
    all_events = drain(engine.events)

    started = [e for e in all_events if isinstance(e, CompactionStarted)]
    assert len(started) == 1
    # Not all 18 old messages were summarised — some were pushed to kept
    assert started[0].old_messages < 18
    # More than just 1 turn (2 msgs) kept
    assert started[0].kept_messages > 2


def test_compact_single_message_exceeds_context(
    config_path: Path, engine: Engine, llm_ctx: LLMContext
) -> None:
    """When even one message exceeds available context, warns gracefully."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    _seed(
        engine,
        [
            _user("x" * 100_000),  # ~25k tokens
            _assistant("ok"),
            _user("next"),
            _assistant("done"),
        ],
    )
    # Context window so small that even 1 message doesn't fit
    engine.state.usage.context_window = 17_000  # 17k - 16k reserve = 1k available

    compact_history(llm_ctx)
    texts = output_texts(drain(engine.events))
    assert any("even a single message exceeds" in t for t in texts)


def test_compact_llm_error_leaves_history_unchanged(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """If the LLM call fails, history is not modified."""

    async def _fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ModelHTTPError(status_code=500, model_name="test", body=b"server error")
        yield ""  # make it a generator

    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    original = list(_turns(15))
    _seed(engine, original)
    engine.state.usage.context_window = 200_000

    stub_provider.set_model(FunctionModel(stream_function=_fail))
    compact_history(llm_ctx)

    # DB unchanged after error — load_messages returns same count.
    assert len(engine.store.load_messages(engine.state.session_id)) == len(original)

    all_events = drain(engine.events)

    # Error event emitted.
    error_outputs = [
        e for e in all_events if isinstance(e, Output) and e.level == OutputLevel.ERROR
    ]
    assert any("Compaction failed" in e.text for e in error_outputs)

    # CompactionStarted and CompactionFinished always paired — even on error.
    assert has_event_type(all_events, CompactionStarted)
    assert has_event_type(all_events, CompactionFinished)
    finished = [e for e in all_events if isinstance(e, CompactionFinished)]
    assert finished[0].summary_tokens == 0


def test_compact_leaves_last_input_tokens_unchanged(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """After compaction, last_input_tokens is unchanged — corrected on next LLM call."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    _seed(engine, _turns(15))
    engine.state.usage.context_window = 200_000
    engine.state.usage.last_input_tokens = 150_000  # simulate high usage

    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(llm_ctx)

    # last_input_tokens untouched — no inaccurate estimate
    assert engine.state.usage.last_input_tokens == 150_000


def test_compact_with_command_inputs(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """Compaction with interleaved command/shell inputs compacts the right messages.

    Command and shell inputs produce rows in the DB that are NOT
    returned by `load_messages`.  Compaction uses
    `load_messages_with_ids` to get paired (id, message) tuples
    from a single query — no index-alignment risk.
    """
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    engine.state.model_name = "claude/sonnet"
    engine.state.usage.context_window = 200_000

    # Seed: 5 turns with command inputs interleaved.
    for i in range(5):
        _seed(engine, [_user(f"question {i}"), _assistant(f"answer {i}")])
        engine.store.save_input(engine.state.session_id, "/help", "command")

    # Command inputs are in the DB but not in load_messages.
    all_messages = engine.store.load_messages(engine.state.session_id)
    assert len(all_messages) == 10  # 5 user + 5 assistant

    # Paired method returns the same 10 messages with their IDs.
    paired = engine.store.load_messages_with_ids(engine.state.session_id)
    assert len(paired) == 10
    assert all(isinstance(mid, str) and len(mid) > 0 for mid, _ in paired)

    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(llm_ctx)

    # After compaction, kept messages are valid — no orphaned parts.
    kept = engine.store.load_messages(engine.state.session_id)
    assert len(kept) >= 3  # at least summary + 1 kept turn


# ═══════════════════════════════════════════════════════════════════════
# Mid-turn compaction — via handle_llm event contract
# ═══════════════════════════════════════════════════════════════════════

# These tests exercise mid-turn compaction through the public
# `run_task(TaskType.LLM, ...)` entry point.  The agent loop runs for real
# with a `FunctionModel` that always calls tools on the first
# request (regardless of history — unlike `TestModel`).  The
# compaction LLM call (`_stream_summary`) is mocked.


def _seed_llm_history(engine: Engine, *, turns: int = 5) -> None:
    """Seed the DB with *turns* of history."""
    msgs = _turns(turns)
    engine._sync_store_context()
    engine.store.save_messages(engine.state.session_id, msgs)


@pytest.fixture
def mid_turn_engine(llm_engine: Engine, stub_provider: StubProvider) -> Engine:
    """Engine configured to trigger mid-turn compaction.

    Uses a tool-calling FunctionModel and a tiny context window
    so compaction triggers after the first tool-call response.
    """
    stub_provider.set_model(_tool_then_text_model())
    _seed_llm_history(llm_engine)
    llm_engine.state.usage.context_window = 100
    llm_engine.state.usage.context_window_known = True
    config.compaction.auto_compact_pct = 1  # trigger on any usage
    return llm_engine


def test_mid_turn_compaction_fires(mid_turn_engine: Engine) -> None:
    """When context exceeds threshold during a tool-call turn,
    compaction fires mid-turn and the model continues.
    """

    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        mid_turn_engine.run_task(TaskType.LLM, "trigger tools")
    events = drain(mid_turn_engine.events)

    assert any(isinstance(e, TaskFinished) and e.success for e in events)
    assert any(isinstance(e, CompactionStarted) for e in events)
    assert any(isinstance(e, CompactionFinished) for e in events)

    # Verify DB history is structurally valid after mid-turn compaction.
    loaded = mid_turn_engine.store.load_messages(mid_turn_engine.state.session_id)
    assert_ordering(loaded)


def test_mid_turn_compaction_produces_summary_in_db(mid_turn_engine: Engine) -> None:
    """After mid-turn compaction the DB contains a summary message."""

    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        mid_turn_engine.run_task(TaskType.LLM, "trigger tools")
    drain(mid_turn_engine.events)

    loaded = mid_turn_engine.store.load_messages(mid_turn_engine.state.session_id)
    assert any(
        isinstance(msg, ModelRequest)
        and any(
            isinstance(p, UserPromptPart) and isinstance(p.content, str) and "Summary" in p.content
            for p in msg.parts
        )
        for msg in loaded
    )

    # Verify DB history is structurally valid after mid-turn compaction.
    loaded = mid_turn_engine.store.load_messages(mid_turn_engine.state.session_id)
    assert_ordering(loaded)


def test_mid_turn_compaction_only_once_per_turn(mid_turn_engine: Engine) -> None:
    """Mid-turn compaction fires at most once — the resume run does not
    re-trigger even if context is still high.
    """

    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        mid_turn_engine.run_task(TaskType.LLM, "trigger tools")
    events = drain(mid_turn_engine.events)

    compaction_count = sum(1 for e in events if isinstance(e, CompactionFinished))
    assert compaction_count == 1

    # Verify DB history is structurally valid after mid-turn compaction.
    loaded = mid_turn_engine.store.load_messages(mid_turn_engine.state.session_id)
    assert_ordering(loaded)


def test_mid_turn_compaction_preserves_continuity(
    mid_turn_engine: Engine, stub_provider: StubProvider
) -> None:
    """After mid-turn compaction the conversation can continue normally."""

    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        mid_turn_engine.run_task(TaskType.LLM, "trigger tools")
    drain(mid_turn_engine.events)

    # Reset to normal context window and send a follow-up turn.
    mid_turn_engine.state.usage.context_window = 200_000
    stub_provider.set_model(TestModel(custom_output_text="follow up response"))
    mid_turn_engine.run_task(TaskType.LLM, "follow up")
    events = drain(mid_turn_engine.events)

    assert any(isinstance(e, TaskFinished) and e.success for e in events)
    loaded = mid_turn_engine.store.load_messages(mid_turn_engine.state.session_id)
    user_texts = [
        p.content
        for msg in loaded
        if isinstance(msg, ModelRequest)
        for p in msg.parts
        if isinstance(p, UserPromptPart) and isinstance(p.content, str)
    ]
    assert "follow up" in user_texts

    # Verify DB history is structurally valid after mid-turn compaction.
    loaded = mid_turn_engine.store.load_messages(mid_turn_engine.state.session_id)
    assert_ordering(loaded)


def test_no_mid_turn_compaction_without_tools(
    config_path: Path, mocker: MockerFixture, mid_turn_engine: Engine, stub_provider: StubProvider
) -> None:
    """When the turn has no tool calls, mid-turn compaction does not
    fire — it falls through to post-turn compaction.
    """
    _seed_llm_history(mid_turn_engine)
    stub_provider.set_model(TestModel(custom_output_text="no tools", call_tools=[]))
    mid_turn_engine.state.usage.context_window = 50
    mid_turn_engine.state.usage.context_window_known = True

    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        mid_turn_engine.run_task(TaskType.LLM, "no tools")
    events = drain(mid_turn_engine.events)
    texts = output_texts(events)

    assert not any("mid-turn" in t.lower() for t in texts)

    # DB history is still valid.
    loaded = mid_turn_engine.store.load_messages(mid_turn_engine.state.session_id)
    assert_ordering(loaded)


def test_mid_turn_compaction_blocks_reset(mid_turn_engine: Engine) -> None:
    """`/compact reset` is blocked after mid-turn compaction because
    the model continues and adds messages after the summary.
    """

    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        mid_turn_engine.run_task(TaskType.LLM, "trigger tools")
    drain(mid_turn_engine.events)

    # The model resumed after mid-turn compaction, so messages
    # exist with IDs > summary_id.  Reset must be blocked.
    reset_compaction(mid_turn_engine._llm_context())
    events = drain(mid_turn_engine.events)
    texts = output_texts(events)

    assert any("Cannot reset" in t for t in texts)

    # ═══════════════════════════════════════════════════════════════════════
    # /compact reset
    # ═══════════════════════════════════════════════════════════════════════

    # Verify DB history is structurally valid after mid-turn compaction.
    loaded = mid_turn_engine.store.load_messages(mid_turn_engine.state.session_id)
    assert_ordering(loaded)


def test_compact_reset_restores_messages(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """`/compact reset` un-marks compacted messages, summary stays."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    original = _turns(8)
    _seed(engine, original)
    engine.state.usage.context_window = 200_000

    # Compact — some messages now marked.
    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(llm_ctx)
    drain(engine.events)
    compacted_count = len(engine.store.load_messages(engine.state.session_id))
    assert compacted_count < len(original)

    # Reset.
    reset_compaction(llm_ctx)
    events = drain(engine.events)
    texts = output_texts(events)

    assert any("reset" in t.lower() and "restored" in t.lower() for t in texts)

    # All original messages are active again, summary deleted.
    restored = engine.store.load_messages(engine.state.session_id)
    assert len(restored) == len(original)
    assert_ordering(restored)
    # Summary is gone.
    assert not any(
        isinstance(m, ModelRequest)
        and any(isinstance(p, UserPromptPart) and SUMMARY_MARKER in str(p.content) for p in m.parts)
        for m in restored
    )
    # Original content recovered — first prompt matches.
    first_req = restored[0]
    assert isinstance(first_req, ModelRequest)
    assert any(isinstance(p, UserPromptPart) and p.content == "question 0" for p in first_req.parts)


def test_compact_reset_no_existing_compaction(
    config_path: Path, engine: Engine, llm_ctx: LLMContext
) -> None:
    """`/compact reset` with no prior compaction says nothing to reset."""
    _seed(engine, _turns(3))
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"

    reset_compaction(llm_ctx)
    events = drain(engine.events)
    texts = output_texts(events)

    assert any("Nothing to reset" in t for t in texts)


def test_compact_reset_only_latest(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """`/compact reset` undoes only the latest compaction, not all."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    engine.state.usage.context_window = 200_000

    # Build history with many turns so two compactions can stack.
    _seed(engine, _turns(20))

    # First compaction.
    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(llm_ctx)
    drain(engine.events)
    after_first = len(engine.store.load_messages(engine.state.session_id))

    # Add more turns, then compact again.
    _seed(engine, _turns(10))
    compact_history(llm_ctx)
    drain(engine.events)
    after_second = len(engine.store.load_messages(engine.state.session_id))
    assert after_second <= after_first + 20  # sanity

    # Reset undoes only the second compaction.
    reset_compaction(llm_ctx)
    drain(engine.events)
    after_reset = len(engine.store.load_messages(engine.state.session_id))

    # More messages than after second compact (restored some).
    assert after_reset > after_second

    # First summary survives, second (the one we reset) is deleted.
    summaries = [
        m
        for m in engine.store.load_messages(engine.state.session_id)
        if isinstance(m, ModelRequest)
        and any(isinstance(p, UserPromptPart) and SUMMARY_MARKER in str(p.content) for p in m.parts)
    ]
    assert len(summaries) == 1

    # First compaction's marks still hold — total active count is
    # less than all 60 original messages (20 + 10 turns x 2 msgs).
    assert after_reset < 60


def test_compact_reset_blocked_after_new_messages(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """`/compact reset` is blocked when messages were added after compaction."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    _seed(engine, _turns(8))
    engine.state.usage.context_window = 200_000

    # Compact.
    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(llm_ctx)
    drain(engine.events)

    # Add new messages after compaction.
    _seed(engine, _turns(2))

    # Reset should be blocked.
    reset_compaction(llm_ctx)
    events = drain(engine.events)
    texts = output_texts(events)

    assert any("Cannot reset" in t for t in texts)

    # Messages are unchanged — compaction was not undone.
    msgs = engine.store.load_messages(engine.state.session_id)
    assert any(
        isinstance(m, ModelRequest)
        and any(isinstance(p, UserPromptPart) and SUMMARY_MARKER in str(p.content) for p in m.parts)
        for m in msgs
    )


def test_compact_reset_allowed_immediately_after_compaction(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """`/compact reset` works when no messages were added after compaction."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    _seed(engine, _turns(8))
    engine.state.usage.context_window = 200_000

    # Compact then immediately reset — no new messages in between.
    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(llm_ctx)
    drain(engine.events)

    reset_compaction(llm_ctx)
    events = drain(engine.events)
    texts = output_texts(events)

    assert any("restored" in t.lower() for t in texts)
    assert not any("Cannot" in t for t in texts)


# ═══════════════════════════════════════════════════════════════════════
# End-to-end: compaction across tool-call boundaries
# ═══════════════════════════════════════════════════════════════════════

# Realistic conversation where tool returns straddle turn boundaries.
# This is the pattern that caused the production bug: a ModelResponse
# at the end of turn N contains a ToolCallPart, but pydantic-ai puts
# the ToolReturnPart in a separate ModelRequest (no UserPromptPart)
# that lands AFTER the next turn's user prompt.  When compaction
# splits at the user-prompt boundary, the tool call goes to old
# but the tool return stays in kept — an orphan.
#
#   Turn 1: user → call_AAA → return_AAA → text
#   Turn 2: user → call_BBB → return_BBB → call_CCC → return_CCC → text
#   Turn 3: response with call_DDD (end of model's multi-step)
#            ↓ user prompt starts turn 4 HERE
#   Turn 4: user → text
#            return_DDD ← orphan! (tool call is in turn 3, above the cut)
#   Turn 5: user → text
#
# With keep_turns=2, the cut falls before turn 4.  The response
# containing call_DDD is in old, but return_DDD is in kept.


def _assert_valid_history(messages: list[ModelMessage]) -> None:
    """Assert messages form a structurally valid LLM conversation.

    Checks:
    - Non-empty, starts with ModelRequest, alternates request/response
    - Every ToolReturnPart has a matching ToolCallPart (no orphaned returns)
    - Every ToolCallPart has a matching ToolReturnPart (no orphaned calls)
    - Tool call always precedes its matching return (correct ordering)
    """
    assert messages, "History is empty"
    assert isinstance(messages[0], ModelRequest), "History must start with ModelRequest"

    # A ModelResponse must always be followed by a ModelRequest.
    # Consecutive ModelRequests are allowed (e.g. tool-return-only
    # request after a request with a user prompt).
    for i in range(1, len(messages)):
        if isinstance(messages[i], ModelResponse) and isinstance(messages[i - 1], ModelResponse):
            raise AssertionError(f"Consecutive ModelResponse at index {i - 1} and {i}")

    # Collect all tool call IDs with their position.
    call_positions: dict[str, int] = {}
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelResponse):
            for resp_part in msg.parts:
                if isinstance(resp_part, ToolCallPart) and resp_part.tool_call_id:
                    call_positions[resp_part.tool_call_id] = i

    # Collect all tool return IDs with their position.
    return_positions: dict[str, int] = {}
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            for req_part in msg.parts:
                if isinstance(req_part, ToolReturnPart) and req_part.tool_call_id:
                    return_positions[req_part.tool_call_id] = i

    # Every return has a matching call.
    for call_id, pos in return_positions.items():
        assert call_id in call_positions, f"Orphaned tool return at message {pos}: {call_id}"

    # Every call has a matching return.
    for call_id, pos in call_positions.items():
        assert call_id in return_positions, f"Orphaned tool call at message {pos}: {call_id}"

    # Call always precedes its return.
    for call_id in call_positions:
        assert call_positions[call_id] < return_positions[call_id], (
            f"Tool call {call_id} at message {call_positions[call_id]} "
            f"appears after its return at message {return_positions[call_id]}"
        )


def _assert_loaded_valid(engine: Engine) -> None:
    """Assert load_messages — with transient repair — returns valid history."""

    loaded, _ = strip_orphaned_tool_returns(engine.store.load_messages(engine.state.session_id))
    _assert_valid_history(loaded)


@parametrize_with_cases("boundary_history", cases=[case_tool_boundary_straddle])
def test_compaction_across_tool_boundaries_no_orphans(
    boundary_history: list[ModelMessage],
    config_path: Path,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """Compacting a conversation where a tool return straddles the
    turn boundary produces no orphaned tool returns.

    With keep_turns=2 the cut falls between turn 3 (which has the
    ToolCallPart for call_DDD) and turn 4 (after which the
    ToolReturnPart for call_DDD appears).  Without the orphan fix,
    call_DDD's return would be in kept with no matching call.
    """
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    engine.state.usage.context_window = 200_000

    _seed(engine, list(boundary_history))
    with compact_agent.override(model=TestModel(custom_output_text="Tool boundary summary.")):
        compact_history(llm_ctx)  # keep_turns=2 from config
    drain(engine.events)
    _assert_loaded_valid(engine)

    # Summary text is in the compacted history.
    loaded = engine.store.load_messages(engine.state.session_id)
    assert any(
        isinstance(p, UserPromptPart) and "Tool boundary summary" in str(p.content)
        for msg in loaded
        if isinstance(msg, ModelRequest)
        for p in msg.parts
    )


@parametrize_with_cases("boundary_history", cases=[case_tool_boundary_straddle])
def test_compact_reset_restores_original_messages_without_summary(
    boundary_history: list[ModelMessage],
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """After `/compact reset`, loaded messages are exactly the
    originals — no summary injected, no orphaned tool returns,
    no interleaving artifacts.
    """
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    engine.state.usage.context_window = 200_000
    _seed(engine, list(boundary_history))

    # Snapshot original messages before compaction.
    original = engine.store.load_messages(engine.state.session_id)
    assert len(original) == len(boundary_history)

    # Compact.
    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(llm_ctx)
    drain(engine.events)
    compacted = engine.store.load_messages(engine.state.session_id)
    assert len(compacted) < len(original)

    # Reset.
    reset_compaction(llm_ctx)
    drain(engine.events)
    restored = engine.store.load_messages(engine.state.session_id)

    # Same count as original — summary is gone.
    assert len(restored) == len(original)

    # No summary marker anywhere.
    for msg in restored:
        if isinstance(msg, ModelRequest):
            for p in msg.parts:
                if isinstance(p, UserPromptPart):
                    assert SUMMARY_MARKER not in str(p.content)

    # No orphaned tool returns.
    _assert_loaded_valid(engine)

    # Message types match original order.
    for orig, rest in zip(original, restored, strict=True):
        assert type(orig) is type(rest)


@parametrize_with_cases("boundary_history", cases=[case_tool_boundary_straddle])
def test_compaction_reset_and_recompact_no_orphans(
    boundary_history: list[ModelMessage],
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
    llm_ctx: LLMContext,
    stub_provider: StubProvider,
) -> None:
    """After reset and recompaction, no orphaned tool returns exist.

    1. Compact (may produce orphans in old code, shouldn't now)
    2. Reset — restores all messages
    3. Recompact — should still produce clean history
    """
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    engine.state.usage.context_window = 200_000
    _seed(engine, list(boundary_history))

    # First compaction.
    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(llm_ctx)
    drain(engine.events)
    _assert_loaded_valid(engine)
    after_compact = len(engine.store.load_messages(engine.state.session_id))

    # Reset — restores originals, deletes summary.
    reset_compaction(llm_ctx)
    drain(engine.events)
    after_reset = engine.store.load_messages(engine.state.session_id)
    assert len(after_reset) == len(boundary_history)
    _assert_loaded_valid(engine)

    # Recompact.
    compact_history(llm_ctx)
    drain(engine.events)
    _assert_loaded_valid(engine)
    after_recompact = len(engine.store.load_messages(engine.state.session_id))
    assert after_recompact <= after_compact
