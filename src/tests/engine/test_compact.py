"""Tests for context compaction — history helpers and engine integration.

Pure functions are tested with realistic message data.  Integration
tests go through ``compact_history`` and check emitted events + session
state, mocking only the LLM call boundary (``_run_summary``).
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from rbtr.engine.compact import find_fit_count
from rbtr.engine.history import (
    _SUMMARY_MARKER,
    build_summary_message,
    estimate_tokens,
    serialise_for_summary,
    split_history,
)
from rbtr.events import CompactionFinished, CompactionStarted, Output

from .conftest import drain, has_event_type, make_engine, output_texts

# ── Test data builders ───────────────────────────────────────────────

_USAGE = RequestUsage(input_tokens=0, output_tokens=0)


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)], usage=_USAGE, model_name="test")


def _tool_return(name: str, content: str) -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name=name, content=content)])


def _tool_call(name: str, args: dict[str, str]) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=name, args=args)], usage=_USAGE, model_name="test"
    )


def _thinking(text: str) -> ModelResponse:
    return ModelResponse(parts=[ThinkingPart(content=text)], usage=_USAGE, model_name="test")


def _turns(n: int) -> list[ModelRequest | ModelResponse]:
    """Create *n* user→assistant turn pairs."""
    msgs: list[ModelRequest | ModelResponse] = []
    for i in range(n):
        msgs.append(_user(f"question {i}"))
        msgs.append(_assistant(f"answer {i}"))
    return msgs


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
# compact_history — integration via event contract
# ═══════════════════════════════════════════════════════════════════════


def test_compact_no_llm(config_path: str) -> None:
    """Warns when no LLM is connected."""
    engine, events, _session = make_engine()

    from rbtr.engine.compact import compact_history

    compact_history(engine)
    texts = output_texts(drain(events))
    assert any("No LLM connected" in t for t in texts)


def test_compact_single_turn(config_path: str) -> None:
    """Single-turn history has nothing to compact."""
    engine, events, session = make_engine()
    session.claude_connected = True
    session.message_history = list(_turns(1))

    from rbtr.engine.compact import compact_history

    compact_history(engine)
    texts = output_texts(drain(events))
    assert any("Nothing to compact" in t for t in texts)


def test_compact_fewer_turns_than_keep_falls_back(config_path: str, mocker: object) -> None:
    """With 3 turns (< keep_turns=10), falls back to keeping 1 turn."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._run_summary",
        return_value="Summary of turns 1 and 2.",
    )

    engine, events, session = make_engine()
    session.claude_connected = True
    session.message_history = list(_turns(3))
    session.usage.context_window = 200_000

    from rbtr.engine.compact import compact_history

    compact_history(engine)
    all_events = drain(events)

    started = [e for e in all_events if isinstance(e, CompactionStarted)]
    assert len(started) == 1
    assert started[0].old_messages == 4  # 2 turns compacted
    assert started[0].kept_messages == 2  # 1 turn kept


def test_compact_replaces_history(config_path: str, mocker: object) -> None:
    """After compaction, history = [summary_msg] + kept turns."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._run_summary",
        return_value="Reviewed PR #42. Found unused import in foo.py.",
    )

    engine, _events, session = make_engine()
    session.claude_connected = True
    session.message_history = list(REALISTIC_HISTORY)
    session.usage.context_window = 200_000

    from rbtr.engine.compact import compact_history

    compact_history(engine)

    # First message is the summary
    first = session.message_history[0]
    assert isinstance(first, ModelRequest)
    part = first.parts[0]
    assert isinstance(part, UserPromptPart)
    assert isinstance(part.content, str)
    assert _SUMMARY_MARKER in part.content
    assert "Reviewed PR #42" in part.content

    # History is shorter than before
    assert len(session.message_history) < len(REALISTIC_HISTORY)

    # Last message is preserved (the last assistant response)
    last = session.message_history[-1]
    assert isinstance(last, ModelResponse)
    assert any(isinstance(p, TextPart) and "import os" in p.content for p in last.parts)


def test_compact_emits_both_events(config_path: str, mocker: object) -> None:
    """Both CompactionStarted and CompactionFinished are emitted."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._run_summary",
        return_value="Summary.",
    )

    engine, events, session = make_engine()
    session.claude_connected = True
    session.message_history = list(_turns(15))
    session.usage.context_window = 200_000

    from rbtr.engine.compact import compact_history

    compact_history(engine)
    all_events = drain(events)
    assert has_event_type(all_events, CompactionStarted)
    assert has_event_type(all_events, CompactionFinished)

    finished = [e for e in all_events if isinstance(e, CompactionFinished)]
    assert finished[0].summary_tokens > 0


def test_compact_extra_instructions_in_prompt(config_path: str, mocker: object) -> None:
    """Extra instructions appear in the prompt sent to the model."""
    mock = mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._run_summary",
        return_value="Summary.",
    )

    engine, _events, session = make_engine()
    session.claude_connected = True
    session.message_history = list(_turns(15))
    session.usage.context_window = 200_000

    from rbtr.engine.compact import compact_history

    compact_history(engine, extra_instructions="Focus on security")

    prompt = mock.call_args[0][2]  # (engine, model, prompt)
    assert "Focus on security" in prompt


def test_compact_over_limit_shrinks_old(config_path: str, mocker: object) -> None:
    """When serialised old exceeds context, only a fitting prefix is summarised."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._run_summary",
        return_value="Partial summary.",
    )

    engine, events, session = make_engine()
    session.claude_connected = True
    # 10 turns with large user messages (each ~2500 tokens after 4-char heuristic)
    big_history: list[ModelRequest | ModelResponse] = []
    for i in range(10):
        big_history.append(_user(f"{'x' * 10_000} question {i}"))
        big_history.append(_assistant(f"answer {i}"))
    session.message_history = list(big_history)
    # Context window that can't fit all old messages after reserve.
    # 10 turns, keep 1 → 9 turns old. Each turn serialises to ~2500 tokens.
    # 9 turns ≈ 22.5k tokens. Set available to ~7.5k so only ~3 turns fit.
    session.usage.context_window = 23_500  # minus 16k reserve = 7.5k available

    from rbtr.engine.compact import compact_history

    compact_history(engine)
    all_events = drain(events)

    started = [e for e in all_events if isinstance(e, CompactionStarted)]
    assert len(started) == 1
    # Not all 18 old messages were summarised — some were pushed to kept
    assert started[0].old_messages < 18
    # More than just 1 turn (2 msgs) kept
    assert started[0].kept_messages > 2


def test_compact_single_message_exceeds_context(config_path: str) -> None:
    """When even one message exceeds available context, warns gracefully."""
    engine, events, session = make_engine()
    session.claude_connected = True
    session.message_history = [
        _user("x" * 100_000),  # ~25k tokens
        _assistant("ok"),
        _user("next"),
        _assistant("done"),
    ]
    # Context window so small that even 1 message doesn't fit
    session.usage.context_window = 17_000  # 17k - 16k reserve = 1k available

    from rbtr.engine.compact import compact_history

    compact_history(engine)
    texts = output_texts(drain(events))
    assert any("even a single message exceeds" in t for t in texts)


def test_compact_llm_error_leaves_history_unchanged(config_path: str, mocker: object) -> None:
    """If the LLM call fails, history is not modified."""
    from pydantic_ai.exceptions import ModelHTTPError

    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._run_summary",
        side_effect=ModelHTTPError(status_code=500, model_name="test", body=b"server error"),
    )

    engine, events, session = make_engine()
    session.claude_connected = True
    original = list(_turns(15))
    session.message_history = list(original)
    session.usage.context_window = 200_000

    from rbtr.engine.compact import compact_history

    compact_history(engine)

    # History unchanged after error
    assert len(session.message_history) == len(original)
    # Error event emitted
    all_events = drain(events)
    error_outputs = [e for e in all_events if isinstance(e, Output) and "error" in e.style]
    assert any("Compaction failed" in e.text for e in error_outputs)


def test_is_context_overflow_detects_overflow() -> None:
    """Context-overflow errors are correctly identified."""
    from pydantic_ai.exceptions import ModelHTTPError

    from rbtr.engine.llm import _is_context_overflow

    exc = ModelHTTPError(status_code=400, model_name="test", body="context length exceeded")
    assert _is_context_overflow(exc)


def test_is_context_overflow_ignores_non_overflow() -> None:
    """Non-overflow errors are not misidentified."""
    from pydantic_ai.exceptions import ModelHTTPError

    from rbtr.engine.llm import _is_context_overflow

    exc = ModelHTTPError(status_code=400, model_name="test", body="invalid api key")
    assert not _is_context_overflow(exc)


def test_compact_leaves_last_input_tokens_unchanged(config_path: str, mocker: object) -> None:
    """After compaction, last_input_tokens is unchanged — corrected on next LLM call."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._run_summary",
        return_value="Short summary.",
    )

    engine, _events, session = make_engine()
    session.claude_connected = True
    session.message_history = list(_turns(15))
    session.usage.context_window = 200_000
    session.usage.last_input_tokens = 150_000  # simulate high usage

    from rbtr.engine.compact import compact_history

    compact_history(engine)

    # last_input_tokens untouched — no inaccurate estimate
    assert session.usage.last_input_tokens == 150_000
