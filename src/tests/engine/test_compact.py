"""Tests for context compaction — history helpers and engine integration.

Pure functions are tested with realistic message data.  Integration
tests go through ``compact_history`` and check emitted events + state,
mocking only the LLM call boundary (``_stream_summary``).
"""

from __future__ import annotations

import pytest
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
from pydantic_ai.usage import RequestUsage, RunUsage

from rbtr.engine import Engine
from rbtr.events import CompactionFinished, CompactionStarted, Output, OutputLevel, TaskFinished
from rbtr.llm.compact import compact_history, find_fit_count, reset_compaction
from rbtr.llm.context import LLMContext
from rbtr.llm.history import (
    _SUMMARY_MARKER,
    build_summary_message,
    estimate_tokens,
    serialise_for_summary,
    snap_to_safe_boundary,
    split_history,
)
from rbtr.providers import BuiltinProvider

from .conftest import (
    _USAGE,
    _assistant,
    _seed,
    _thinking,
    _tool_call,
    _tool_return,
    _turns,
    _user,
    drain,
    has_event_type,
    output_texts,
)

# ── Test data ────────────────────────────────────────────────────────

# A realistic multi-turn conversation with tool calls.
REALISTIC_HISTORY: list[ModelRequest | ModelResponse] = [
    _user("Review PR #42"),
    _assistant("I'll start by looking at the diff."),
    _tool_call("diff", {"base": "main", "head": "feature"}),
    _tool_return("diff", "--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,5 @@\n+import os\n def main():"),
    _assistant("The diff adds an `import os`. Let me check callers."),
    _tool_call("get_callers", {"symbol": "main"}),
    _tool_return("get_callers", "bar.py:10  baz.py:20"),
    _assistant("Two callers found. Both look fine."),
    _user("What about test coverage?"),
    _thinking("Let me search for tests..."),
    _assistant("Let me check the test files."),
    _tool_call("search_codebase", {"query": "test_main"}),
    _tool_return("search_codebase", "test_foo.py:5  test_bar.py:15"),
    _assistant("Tests exist in test_foo.py and test_bar.py. Coverage looks adequate."),
    _user("Any security concerns?"),
    _assistant("The `import os` is unused — it should be removed. No security issues otherwise."),
]


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
    history = [
        _user("q1"),
        _tool_call("read_file", {"path": "a.py"}),
        _tool_return("read_file", "content"),
        _assistant("a1"),
        _user("q2"),
        _assistant("a2"),
    ]
    old, kept = split_history(history, keep_turns=1)
    assert old == history[:4]  # first turn with its tool calls
    assert kept == history[4:]  # second turn only


def test_split_history_realistic() -> None:
    """Split the realistic conversation — 3 user turns, keep 1."""
    old, kept = split_history(REALISTIC_HISTORY, keep_turns=1)
    # Last user turn starts at "Any security concerns?"
    last_user_idx = next(
        i
        for i in reversed(range(len(REALISTIC_HISTORY)))
        if isinstance(REALISTIC_HISTORY[i], ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in REALISTIC_HISTORY[i].parts)
    )
    assert kept == REALISTIC_HISTORY[last_user_idx:]
    assert old == REALISTIC_HISTORY[:last_user_idx]


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
            for p in msg.parts:
                if isinstance(p, ToolCallPart) and p.tool_call_id:
                    call_ids.add(p.tool_call_id)
    for msg in kept:
        if isinstance(msg, ModelRequest):
            for p in msg.parts:
                if isinstance(p, ToolReturnPart) and p.tool_call_id:
                    assert p.tool_call_id in call_ids, f"Orphaned tool return: {p.tool_call_id}"


def test_orphan_single_tool_return_at_boundary() -> None:
    """A lone tool return right after the cut point is moved to old.

    Turn 1 ends with a tool call; its return sits just before turn 2.
    Keeping only turn 2 must move the orphaned return into old.
    """
    history: list[ModelMessage] = [
        _user("turn 1"),  # [0]
        _tool_call("grep", call_id="call_A"),  # [1]
        _tool_return("grep", "result", call_id="call_A"),  # [2] — orphan candidate
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
        _tool_return("grep", "result", call_id="call_A"),
        _tool_return("read", "result", call_id="call_B"),
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
        _tool_call("grep", call_id="call_A"),
        _tool_return("grep", "result", call_id="call_A"),
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
        _tool_call("grep", call_id="call_A"),
        _tool_return("grep", "result", call_id="call_A"),
        _assistant("answer 1"),
        _user("turn 2"),
        _tool_call("read", call_id="call_B"),
        _tool_return("read", "result", call_id="call_B"),
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
        _tool_call("grep", call_id="call_A"),
        _tool_return("grep", "result", call_id="call_A"),
        _assistant("answer 1"),
        _user("turn 2"),
        _tool_call("read", call_id="call_B"),
        _tool_return("read", "result", call_id="call_B"),
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
        _tool_call("grep", call_id="call_A"),
        _tool_return("grep", "result", call_id="call_A"),
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
        ModelRequest(parts=[ToolReturnPart(tool_name="grep", content="x", tool_call_id=None)]),
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
        _tool_call("grep", call_id="call_A"),
        _tool_return("grep", "result", call_id="call_A"),
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
        _tool_call("grep", call_id="call_A"),
        _tool_return("grep", "result", call_id="call_A"),
        _tool_call("read", call_id="call_B"),
        _tool_return("read", "result", call_id="call_B"),
        _assistant("done"),
    ]
    _old, kept = split_history(history, keep_turns=1)
    assert kept == history[2:]
    _assert_no_orphaned_returns(kept)


def test_orphan_all_messages_kept_no_change() -> None:
    """When nothing is compacted, no orphan cleanup is needed."""
    history: list[ModelMessage] = [
        _user("turn 1"),
        _tool_call("grep", call_id="call_A"),
        _tool_return("grep", "result", call_id="call_A"),
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
        _tool_call("diff", call_id="call_111"),
        _tool_return("diff", "result", call_id="call_111"),
        _assistant("Here's the diff."),
        _tool_call("list_files", call_id="call_xJJ"),
        # ↑ This response has the tool call for list_files.
        # ↓ The user prompt starts a new turn, splitting here.
        _user("check notes"),
        _assistant("No notes found."),
        # The tool return for call_xJJ arrives AFTER the user prompt
        # (pydantic-ai structures it as a separate ModelRequest).
        _tool_return("list_files", "result", call_id="call_xJJ"),
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


def test_serialise_realistic_conversation() -> None:
    """Serialise the full realistic history and check all section types."""
    result = serialise_for_summary(REALISTIC_HISTORY)
    assert "## User\nReview PR #42" in result
    assert "## Assistant\nI'll start by looking at the diff." in result
    assert '## Tool call: diff({"base": "main", "head": "feature"})' in result
    assert "## Tool result: diff\n--- a/foo.py" in result
    # ThinkingPart should be omitted
    assert "Let me search for tests" not in result


def test_serialise_truncates_tool_results() -> None:
    """Tool results exceeding max_tool_chars are truncated."""
    long_diff = "x" * 5_000
    messages = [_tool_return("diff", long_diff)]
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
    assert _SUMMARY_MARKER in part.content
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
    messages = [_user("x" * 10_000)]  # ~2500 tokens
    assert find_fit_count(messages, available_tokens=100, max_tool_chars=2_000) == 0


def test_find_fit_count_partial() -> None:
    """Returns the largest prefix that fits."""
    # Each user message is ~250 tokens (1000 chars)
    messages = [_user("x" * 1000) for _ in range(10)]
    # Budget for ~3 messages worth (header + content ≈ 260 tokens each)
    serialised_3 = serialise_for_summary(messages[:3], max_tool_chars=2_000)
    budget = estimate_tokens(serialised_3)
    result = find_fit_count(messages, available_tokens=budget, max_tool_chars=2_000)
    assert result == 3


def test_find_fit_count_with_tool_results() -> None:
    """Tool result truncation affects what fits."""
    messages: list[ModelRequest | ModelResponse] = [
        _tool_return("diff", "x" * 8_000),
        _tool_return("read_file", "y" * 8_000),
        _tool_return("search", "z" * 8_000),
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
        _tool_call("read_file", {"path": "a.py"}, call_id="tc1"),
        _tool_return("read_file", "contents", call_id="tc1"),
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
        _tool_call("grep", call_id="tc1"),
        _tool_return("grep", "result", call_id="tc1"),
        _tool_call("diff", call_id="tc2"),
        _tool_return("diff", "result", call_id="tc2"),
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
        _tool_call("read_file", call_id="tc1"),
        _tool_return("read_file", "contents", call_id="tc1"),
    ]
    assert snap_to_safe_boundary(messages, 1) == 0


def test_snap_zero_stays_zero() -> None:
    """count=0 is always returned as-is."""
    messages: list[ModelMessage] = [_tool_call("read_file", call_id="tc1")]
    assert snap_to_safe_boundary(messages, 0) == 0


def test_snap_text_response_is_safe() -> None:
    """A ModelResponse with only text (no tool calls) is a safe boundary."""
    messages: list[ModelMessage] = [
        _user("hello"),
        _assistant("thinking..."),
        _tool_call("grep", call_id="tc1"),
        _tool_return("grep", "found", call_id="tc1"),
    ]
    # count=2 → after _assistant (text only) → safe
    assert snap_to_safe_boundary(messages, 2) == 2


def test_snap_realistic_mid_turn() -> None:
    """Realistic mid-turn pattern: multiple tool rounds in one turn."""
    messages: list[ModelMessage] = [
        _user("review PR"),  # 0
        _tool_call("diff", call_id="tc1"),  # 1
        _tool_return("diff", "...", call_id="tc1"),  # 2
        _tool_call("read_file", call_id="tc2"),  # 3
        _tool_return("read_file", "...", call_id="tc2"),  # 4
        _tool_call("grep", call_id="tc3"),  # 5
        _tool_return("grep", "...", call_id="tc3"),  # 6
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


def test_compact_no_llm(config_path: str, engine: Engine, llm_ctx: LLMContext) -> None:
    """Warns when no LLM is connected."""

    compact_history(llm_ctx)
    texts = output_texts(drain(engine.events))
    assert any("No LLM connected" in t for t in texts)


def test_compact_single_turn(config_path: str, engine: Engine, llm_ctx: LLMContext) -> None:
    """Single-turn history has nothing to compact."""
    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    _seed(engine, _turns(1))

    compact_history(llm_ctx)
    texts = output_texts(drain(engine.events))
    assert any("Nothing to compact" in t for t in texts)


def test_compact_fewer_turns_than_keep_falls_back(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """With 2 turns (= keep_turns=2), normal split finds nothing to
    compact, so it falls back to keeping 1 turn.
    """
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.llm.compact._stream_summary",
        return_value="Summary of turn 1.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    _seed(engine, _turns(2))
    engine.state.usage.context_window = 200_000

    compact_history(llm_ctx)
    all_events = drain(engine.events)

    started = [e for e in all_events if isinstance(e, CompactionStarted)]
    assert len(started) == 1
    assert started[0].old_messages == 2  # 1 turn compacted
    assert started[0].kept_messages == 2  # 1 turn kept


def test_compact_replaces_history(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """After compaction, history = [summary_msg] + kept turns."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.llm.compact._stream_summary",
        return_value="Reviewed PR #42. Found unused import in foo.py.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    _seed(engine, list(REALISTIC_HISTORY))
    engine.state.usage.context_window = 200_000

    compact_history(llm_ctx)

    # Load from DB — the source of truth.
    loaded = engine.store.load_messages(engine.state.session_id)

    # First message is the summary
    first = loaded[0]
    assert isinstance(first, ModelRequest)
    part = first.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, str)
    assert _SUMMARY_MARKER in part.content
    assert "Reviewed PR #42" in part.content

    # History is shorter than before
    assert len(loaded) < len(REALISTIC_HISTORY)

    # Last message is preserved (the last assistant response)
    last = loaded[-1]
    assert isinstance(last, ModelResponse)
    assert any(isinstance(p, TextPart) and "import os" in p.content for p in last.parts)


def test_compact_emits_both_events(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """Both CompactionStarted and CompactionFinished are emitted."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.llm.compact._stream_summary",
        return_value="Summary.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    _seed(engine, _turns(15))
    engine.state.usage.context_window = 200_000

    compact_history(llm_ctx)
    all_events = drain(engine.events)
    assert has_event_type(all_events, CompactionStarted)
    assert has_event_type(all_events, CompactionFinished)

    finished = [e for e in all_events if isinstance(e, CompactionFinished)]
    assert finished[0].summary_tokens > 0


def test_compact_extra_instructions_in_prompt(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """Extra instructions appear in the prompt sent to the model."""
    mock = mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.llm.compact._stream_summary",
        return_value="Summary.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    _seed(engine, _turns(15))
    engine.state.usage.context_window = 200_000

    compact_history(llm_ctx, "Focus on security")

    prompt = mock.call_args[0][2]  # (engine, model, prompt)
    assert "Focus on security" in prompt


def test_compact_over_limit_shrinks_old(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """When serialised old exceeds context, only a fitting prefix is summarised."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.llm.compact._stream_summary",
        return_value="Partial summary.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
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

    compact_history(llm_ctx)
    all_events = drain(engine.events)

    started = [e for e in all_events if isinstance(e, CompactionStarted)]
    assert len(started) == 1
    # Not all 18 old messages were summarised — some were pushed to kept
    assert started[0].old_messages < 18
    # More than just 1 turn (2 msgs) kept
    assert started[0].kept_messages > 2


def test_compact_single_message_exceeds_context(
    config_path: str, engine: Engine, llm_ctx: LLMContext
) -> None:
    """When even one message exceeds available context, warns gracefully."""
    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
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
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """If the LLM call fails, history is not modified."""
    from pydantic_ai.exceptions import ModelHTTPError

    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.llm.compact._stream_summary",
        side_effect=ModelHTTPError(status_code=500, model_name="test", body=b"server error"),
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    original = list(_turns(15))
    _seed(engine, original)
    engine.state.usage.context_window = 200_000

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
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """After compaction, last_input_tokens is unchanged — corrected on next LLM call."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.llm.compact._stream_summary",
        return_value="Short summary.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    _seed(engine, _turns(15))
    engine.state.usage.context_window = 200_000
    engine.state.usage.last_input_tokens = 150_000  # simulate high usage

    compact_history(llm_ctx)

    # last_input_tokens untouched — no inaccurate estimate
    assert engine.state.usage.last_input_tokens == 150_000


def test_compact_with_command_inputs(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """Compaction with interleaved command/shell inputs compacts the right messages.

    Command and shell inputs produce rows in the DB that are NOT
    returned by ``load_messages``.  Compaction uses
    ``load_messages_with_ids`` to get paired (id, message) tuples
    from a single query — no index-alignment risk.
    """
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value="Summary of conversation.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
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

    compact_history(llm_ctx)

    # After compaction, kept messages are valid — no orphaned parts.
    kept = engine.store.load_messages(engine.state.session_id)
    assert len(kept) >= 3  # at least summary + 1 kept turn


# ═══════════════════════════════════════════════════════════════════════
# Mid-turn compaction — via handle_llm event contract
# ═══════════════════════════════════════════════════════════════════════

# These tests exercise mid-turn compaction through the public
# ``run_task("llm", ...)`` entry point.  The agent loop runs for real
# with a ``FunctionModel`` that always calls tools on the first
# request (regardless of history — unlike ``TestModel``).  The
# compaction LLM call (``_stream_summary``) is mocked.


def _tool_then_text_model() -> FunctionModel:
    """A ``FunctionModel`` that calls a tool on its first request,
    then returns text on the second.

    Stateful: each instance has its own call counter.  Works
    correctly with message history (unlike ``TestModel``).
    Provides both ``function`` and ``stream_function`` so the
    agent's streaming iteration works.
    """
    from collections.abc import AsyncIterator

    from pydantic_ai.models.function import DeltaToolCall

    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        usage = RequestUsage(input_tokens=50, output_tokens=10)
        if call_count == 1 and info.function_tools:
            tool = info.function_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool_name=tool.name, args="{}")],
                usage=usage,
                model_name="test-fn",
            )
        return ModelResponse(
            parts=[TextPart(content="done")],
            usage=usage,
            model_name="test-fn",
        )

    async def stream_fn(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1 and info.function_tools:
            tool = info.function_tools[0]
            yield {0: DeltaToolCall(name=tool.name, json_args="{}")}
        else:
            yield "done"

    return FunctionModel(model_fn, stream_function=stream_fn)


def _text_only_model() -> FunctionModel:
    """A ``FunctionModel`` that always returns text, never tools."""
    from collections.abc import AsyncIterator

    from pydantic_ai.models.function import DeltaToolCall

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content="text only")],
            usage=RequestUsage(input_tokens=50, output_tokens=10),
            model_name="test-fn",
        )

    async def stream_fn(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        yield "text only"

    return FunctionModel(model_fn, stream_function=stream_fn)


def _seed_llm_history(engine: Engine, *, turns: int = 5) -> None:
    """Seed the DB with *turns* of history."""
    msgs = _turns(turns)
    engine._sync_store_context()
    engine.store.save_messages(engine.state.session_id, msgs)


def _patch_for_mid_turn(engine: Engine, mocker: object) -> object:
    """Patch ``build_model`` with a tool-calling ``FunctionModel`` and
    set up compaction mocks.

    ``_update_live_usage`` is wrapped to shrink the context window
    after each model response, simulating high context usage.

    Returns the ``_stream_summary`` mock for call-count assertions.
    """
    from rbtr.llm.stream import _update_live_usage as _real_update

    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.stream.build_model",
        return_value=_tool_then_text_model(),
    )
    summary_mock = mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary", return_value="Summary."
    )
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact.build_model",
        return_value=_text_only_model(),
    )

    def _inflating_update(eng: Engine, run_usage: object, response: object) -> None:
        _real_update(eng, run_usage, response)  # type: ignore[arg-type]
        eng.state.usage.context_window = 50
        eng.state.usage.context_window_known = True

    mocker.patch("rbtr.llm.stream._update_live_usage", side_effect=_inflating_update)  # type: ignore[union-attr]
    return summary_mock


def test_mid_turn_compaction_fires(config_path: str, mocker: object, llm_engine: Engine) -> None:
    """When context exceeds threshold during a tool-call turn,
    compaction fires mid-turn and the model continues.
    """
    _seed_llm_history(llm_engine)
    _patch_for_mid_turn(llm_engine, mocker)

    llm_engine.run_task("llm", "trigger tools")
    events = drain(llm_engine.events)

    assert any(isinstance(e, TaskFinished) and e.success for e in events)
    assert any(isinstance(e, CompactionStarted) for e in events)
    assert any(isinstance(e, CompactionFinished) for e in events)


def test_mid_turn_compaction_produces_summary_in_db(
    config_path: str, mocker: object, llm_engine: Engine
) -> None:
    """After mid-turn compaction the DB contains a summary message."""
    _seed_llm_history(llm_engine)
    _patch_for_mid_turn(llm_engine, mocker)

    llm_engine.run_task("llm", "trigger tools")
    drain(llm_engine.events)

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert any(
        isinstance(msg, ModelRequest)
        and any(
            isinstance(p, UserPromptPart) and isinstance(p.content, str) and "Summary" in p.content
            for p in msg.parts
        )
        for msg in loaded
    )


def test_mid_turn_compaction_only_once_per_turn(
    config_path: str, mocker: object, llm_engine: Engine
) -> None:
    """Mid-turn compaction fires at most once — the resume run does not
    re-trigger even if context is still high.
    """
    _seed_llm_history(llm_engine)
    summary_mock = _patch_for_mid_turn(llm_engine, mocker)

    llm_engine.run_task("llm", "trigger tools")
    drain(llm_engine.events)

    assert summary_mock.call_count == 1  # type: ignore[union-attr]


def test_mid_turn_compaction_preserves_continuity(
    config_path: str, mocker: object, llm_engine: Engine
) -> None:
    """After mid-turn compaction the conversation can continue normally."""
    _seed_llm_history(llm_engine)
    _patch_for_mid_turn(llm_engine, mocker)

    llm_engine.run_task("llm", "trigger tools")
    drain(llm_engine.events)

    # Reset to normal context window and send a follow-up turn.
    llm_engine.state.usage.context_window = 200_000
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.stream.build_model",
        return_value=_text_only_model(),
    )
    llm_engine.run_task("llm", "follow up")
    events = drain(llm_engine.events)

    assert any(isinstance(e, TaskFinished) and e.success for e in events)
    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    user_texts = [
        p.content
        for msg in loaded
        if isinstance(msg, ModelRequest)
        for p in msg.parts
        if isinstance(p, UserPromptPart) and isinstance(p.content, str)
    ]
    assert "follow up" in user_texts


def test_no_mid_turn_compaction_without_tools(
    config_path: str, mocker: object, llm_engine: Engine
) -> None:
    """When the turn has no tool calls, mid-turn compaction does not
    fire — it falls through to post-turn compaction.
    """
    _seed_llm_history(llm_engine)
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.stream.build_model",
        return_value=_text_only_model(),
    )
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary", return_value="Summary."
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]
    llm_engine.state.usage.context_window = 50
    llm_engine.state.usage.context_window_known = True

    llm_engine.run_task("llm", "no tools")
    events = drain(llm_engine.events)
    texts = output_texts(events)

    assert not any("mid-turn" in t.lower() for t in texts)


def test_mid_turn_compaction_blocks_reset(
    config_path: str, mocker: object, llm_engine: Engine
) -> None:
    """``/compact reset`` is blocked after mid-turn compaction because
    the model continues and adds messages after the summary.
    """
    _seed_llm_history(llm_engine)
    _patch_for_mid_turn(llm_engine, mocker)

    llm_engine.run_task("llm", "trigger tools")
    drain(llm_engine.events)

    # The model resumed after mid-turn compaction, so messages
    # exist with IDs > summary_id.  Reset must be blocked.
    reset_compaction(llm_engine._llm_context())
    events = drain(llm_engine.events)
    texts = output_texts(events)

    assert any("Cannot reset" in t for t in texts)


# ═══════════════════════════════════════════════════════════════════════
# /compact reset
# ═══════════════════════════════════════════════════════════════════════


def test_compact_reset_restores_messages(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """``/compact reset`` un-marks compacted messages, summary stays."""
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value="Summary.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    original = _turns(8)
    _seed(engine, original)
    engine.state.usage.context_window = 200_000

    # Compact — some messages now marked.
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
    # Summary is gone.
    assert not any(
        isinstance(m, ModelRequest)
        and any(
            isinstance(p, UserPromptPart) and _SUMMARY_MARKER in str(p.content) for p in m.parts
        )
        for m in restored
    )


def test_compact_reset_no_existing_compaction(
    config_path: str, engine: Engine, llm_ctx: LLMContext
) -> None:
    """``/compact reset`` with no prior compaction says nothing to reset."""
    _seed(engine, _turns(3))
    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"

    reset_compaction(llm_ctx)
    events = drain(engine.events)
    texts = output_texts(events)

    assert any("Nothing to reset" in t for t in texts)


def test_compact_reset_only_latest(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """``/compact reset`` undoes only the latest compaction, not all."""
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value="Summary.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine.state.usage.context_window = 200_000

    # Build history with many turns so two compactions can stack.
    _seed(engine, _turns(20))

    # First compaction.
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
        and any(
            isinstance(p, UserPromptPart) and _SUMMARY_MARKER in str(p.content) for p in m.parts
        )
    ]
    assert len(summaries) == 1

    # First compaction's marks still hold — total active count is
    # less than all 60 original messages (20 + 10 turns x 2 msgs).
    assert after_reset < 60


def test_compact_reset_blocked_after_new_messages(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """``/compact reset`` is blocked when messages were added after compaction."""
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value="Summary.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    _seed(engine, _turns(8))
    engine.state.usage.context_window = 200_000

    # Compact.
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
        and any(
            isinstance(p, UserPromptPart) and _SUMMARY_MARKER in str(p.content) for p in m.parts
        )
        for m in msgs
    )


def test_compact_reset_allowed_immediately_after_compaction(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """``/compact reset`` works when no messages were added after compaction."""
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value="Summary.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    _seed(engine, _turns(8))
    engine.state.usage.context_window = 200_000

    # Compact then immediately reset — no new messages in between.
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
TOOL_BOUNDARY_HISTORY: list[ModelRequest | ModelResponse] = [
    # Turn 1
    _user("find all TODO comments"),
    _tool_call("grep", {"pattern": "TODO"}, call_id="call_AAA"),
    _tool_return("grep", "foo.py:10 TODO\nbar.py:3 TODO", call_id="call_AAA"),
    _assistant("Found 2 TODOs."),
    # Turn 2
    _user("show me foo.py"),
    _tool_call("read_file", {"path": "foo.py"}, call_id="call_BBB"),
    _tool_return("read_file", "def main(): pass", call_id="call_BBB"),
    _tool_call("grep", {"pattern": "main"}, call_id="call_CCC"),
    _tool_return("grep", "foo.py:1 def main()", call_id="call_CCC"),
    _assistant("foo.py has one function."),
    # Turn 3: model responds with text + tool call in one response
    _user("what other files?"),
    ModelResponse(
        parts=[
            TextPart(content="Let me check."),
            ToolCallPart(tool_name="list_files", args={"path": "."}, tool_call_id="call_DDD"),
        ],
        usage=RunUsage(requests=1),
        model_name="test",
    ),
    # ---- cut falls here with keep_turns=2 ----
    # Turn 4: user prompt starts, but return_DDD arrives after it
    _user("any security concerns?"),
    _assistant("No issues."),
    _tool_return("list_files", "foo.py\nbar.py", call_id="call_DDD"),
    # Turn 5
    _user("summarise"),
    _assistant("All good."),
]


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
            for p in msg.parts:
                if isinstance(p, ToolCallPart) and p.tool_call_id:
                    call_positions[p.tool_call_id] = i

    # Collect all tool return IDs with their position.
    return_positions: dict[str, int] = {}
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            for p in msg.parts:
                if isinstance(p, ToolReturnPart) and p.tool_call_id:
                    return_positions[p.tool_call_id] = i

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
    """Assert load_messages returns a structurally valid conversation."""
    _assert_valid_history(engine.store.load_messages(engine.state.session_id))


def test_compaction_across_tool_boundaries_no_orphans(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """Compacting a conversation where a tool return straddles the
    turn boundary produces no orphaned tool returns.

    With keep_turns=2 the cut falls between turn 3 (which has the
    ToolCallPart for call_DDD) and turn 4 (after which the
    ToolReturnPart for call_DDD appears).  Without the orphan fix,
    call_DDD's return would be in kept with no matching call.
    """
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value="Found TODOs, read foo.py, listed project files.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine.state.usage.context_window = 200_000

    _seed(engine, list(TOOL_BOUNDARY_HISTORY))
    compact_history(llm_ctx)  # keep_turns=2 from config
    drain(engine.events)
    _assert_loaded_valid(engine)


def test_compact_reset_restores_original_messages_without_summary(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """After ``/compact reset``, loaded messages are exactly the
    originals — no summary injected, no orphaned tool returns,
    no interleaving artifacts.
    """
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value="Summary of tool conversation.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine.state.usage.context_window = 200_000
    _seed(engine, list(TOOL_BOUNDARY_HISTORY))

    # Snapshot original messages before compaction.
    original = engine.store.load_messages(engine.state.session_id)
    assert len(original) == len(TOOL_BOUNDARY_HISTORY)

    # Compact.
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
                    assert _SUMMARY_MARKER not in str(p.content)

    # No orphaned tool returns.
    _assert_loaded_valid(engine)

    # Message types match original order.
    for orig, rest in zip(original, restored, strict=True):
        assert type(orig) is type(rest)


def test_compaction_reset_and_recompact_no_orphans(
    config_path: str, mocker: object, engine: Engine, llm_ctx: LLMContext
) -> None:
    """After reset and recompaction, no orphaned tool returns exist.

    1. Compact (may produce orphans in old code, shouldn't now)
    2. Reset — restores all messages
    3. Recompact — should still produce clean history
    """
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value="Summary of tool-heavy conversation.",
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine.state.usage.context_window = 200_000
    _seed(engine, list(TOOL_BOUNDARY_HISTORY))

    # First compaction.
    compact_history(llm_ctx)
    drain(engine.events)
    _assert_loaded_valid(engine)
    after_compact = len(engine.store.load_messages(engine.state.session_id))

    # Reset — restores originals, deletes summary.
    reset_compaction(llm_ctx)
    drain(engine.events)
    after_reset = engine.store.load_messages(engine.state.session_id)
    assert len(after_reset) == len(TOOL_BOUNDARY_HISTORY)
    _assert_loaded_valid(engine)

    # Recompact.
    compact_history(llm_ctx)
    drain(engine.events)
    _assert_loaded_valid(engine)
    after_recompact = len(engine.store.load_messages(engine.state.session_id))
    assert after_recompact <= after_compact
