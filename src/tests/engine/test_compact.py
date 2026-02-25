"""Tests for context compaction — history helpers and engine integration.

Pure functions are tested with realistic message data.  Integration
tests go through ``compact_history`` and check emitted events + state,
mocking only the LLM call boundary (``_stream_summary``).
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from rbtr.engine import Engine
from rbtr.engine.compact import compact_history, find_fit_count
from rbtr.engine.history import (
    _SUMMARY_MARKER,
    build_summary_message,
    estimate_tokens,
    serialise_for_summary,
    split_history,
)
from rbtr.events import CompactionFinished, CompactionStarted, Output, TaskFinished

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


def test_compact_no_llm(config_path: str, engine: Engine) -> None:
    """Warns when no LLM is connected."""

    compact_history(engine)
    texts = output_texts(drain(engine.events))
    assert any("No LLM connected" in t for t in texts)


def test_compact_single_turn(config_path: str, engine: Engine) -> None:
    """Single-turn history has nothing to compact."""
    engine.state.claude_connected = True
    _seed(engine, _turns(1))

    from rbtr.engine.compact import compact_history

    compact_history(engine)
    texts = output_texts(drain(engine.events))
    assert any("Nothing to compact" in t for t in texts)


def test_compact_fewer_turns_than_keep_falls_back(
    config_path: str, mocker: object, engine: Engine
) -> None:
    """With 2 turns (= keep_turns=2), normal split finds nothing to
    compact, so it falls back to keeping 1 turn.
    """
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._stream_summary",
        return_value="Summary of turn 1.",
    )
    mocker.patch("rbtr.engine.compact.build_model")  # type: ignore[union-attr]

    engine.state.claude_connected = True
    _seed(engine, _turns(2))
    engine.state.usage.context_window = 200_000

    compact_history(engine)
    all_events = drain(engine.events)

    started = [e for e in all_events if isinstance(e, CompactionStarted)]
    assert len(started) == 1
    assert started[0].old_messages == 2  # 1 turn compacted
    assert started[0].kept_messages == 2  # 1 turn kept


def test_compact_replaces_history(config_path: str, mocker: object, engine: Engine) -> None:
    """After compaction, history = [summary_msg] + kept turns."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._stream_summary",
        return_value="Reviewed PR #42. Found unused import in foo.py.",
    )
    mocker.patch("rbtr.engine.compact.build_model")  # type: ignore[union-attr]

    engine.state.claude_connected = True
    _seed(engine, list(REALISTIC_HISTORY))
    engine.state.usage.context_window = 200_000

    from rbtr.engine.compact import compact_history

    compact_history(engine)

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


def test_compact_emits_both_events(config_path: str, mocker: object, engine: Engine) -> None:
    """Both CompactionStarted and CompactionFinished are emitted."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._stream_summary",
        return_value="Summary.",
    )
    mocker.patch("rbtr.engine.compact.build_model")  # type: ignore[union-attr]

    engine.state.claude_connected = True
    _seed(engine, _turns(15))
    engine.state.usage.context_window = 200_000

    from rbtr.engine.compact import compact_history

    compact_history(engine)
    all_events = drain(engine.events)
    assert has_event_type(all_events, CompactionStarted)
    assert has_event_type(all_events, CompactionFinished)

    finished = [e for e in all_events if isinstance(e, CompactionFinished)]
    assert finished[0].summary_tokens > 0


def test_compact_extra_instructions_in_prompt(
    config_path: str, mocker: object, engine: Engine
) -> None:
    """Extra instructions appear in the prompt sent to the model."""
    mock = mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._stream_summary",
        return_value="Summary.",
    )
    mocker.patch("rbtr.engine.compact.build_model")  # type: ignore[union-attr]

    engine.state.claude_connected = True
    _seed(engine, _turns(15))
    engine.state.usage.context_window = 200_000

    from rbtr.engine.compact import compact_history

    compact_history(engine, extra_instructions="Focus on security")

    prompt = mock.call_args[0][2]  # (engine, model, prompt)
    assert "Focus on security" in prompt


def test_compact_over_limit_shrinks_old(config_path: str, mocker: object, engine: Engine) -> None:
    """When serialised old exceeds context, only a fitting prefix is summarised."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._stream_summary",
        return_value="Partial summary.",
    )
    mocker.patch("rbtr.engine.compact.build_model")  # type: ignore[union-attr]

    engine.state.claude_connected = True
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

    from rbtr.engine.compact import compact_history

    compact_history(engine)
    all_events = drain(engine.events)

    started = [e for e in all_events if isinstance(e, CompactionStarted)]
    assert len(started) == 1
    # Not all 18 old messages were summarised — some were pushed to kept
    assert started[0].old_messages < 18
    # More than just 1 turn (2 msgs) kept
    assert started[0].kept_messages > 2


def test_compact_single_message_exceeds_context(config_path: str, engine: Engine) -> None:
    """When even one message exceeds available context, warns gracefully."""
    engine.state.claude_connected = True
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

    from rbtr.engine.compact import compact_history

    compact_history(engine)
    texts = output_texts(drain(engine.events))
    assert any("even a single message exceeds" in t for t in texts)


def test_compact_llm_error_leaves_history_unchanged(
    config_path: str, mocker: object, engine: Engine
) -> None:
    """If the LLM call fails, history is not modified."""
    from pydantic_ai.exceptions import ModelHTTPError

    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._stream_summary",
        side_effect=ModelHTTPError(status_code=500, model_name="test", body=b"server error"),
    )
    mocker.patch("rbtr.engine.compact.build_model")  # type: ignore[union-attr]

    engine.state.claude_connected = True
    original = list(_turns(15))
    _seed(engine, original)
    engine.state.usage.context_window = 200_000

    from rbtr.engine.compact import compact_history

    compact_history(engine)

    # DB unchanged after error — load_messages returns same count.
    assert len(engine.store.load_messages(engine.state.session_id)) == len(original)
    # Error event emitted
    all_events = drain(engine.events)
    error_outputs = [e for e in all_events if isinstance(e, Output) and "error" in e.style]
    assert any("Compaction failed" in e.text for e in error_outputs)


def test_compact_leaves_last_input_tokens_unchanged(
    config_path: str, mocker: object, engine: Engine
) -> None:
    """After compaction, last_input_tokens is unchanged — corrected on next LLM call."""
    mocker.patch(  # type: ignore[union-attr]  # mocker is pytest_mock.MockerFixture
        "rbtr.engine.compact._stream_summary",
        return_value="Short summary.",
    )
    mocker.patch("rbtr.engine.compact.build_model")  # type: ignore[union-attr]

    engine.state.claude_connected = True
    _seed(engine, _turns(15))
    engine.state.usage.context_window = 200_000
    engine.state.usage.last_input_tokens = 150_000  # simulate high usage

    from rbtr.engine.compact import compact_history

    compact_history(engine)

    # last_input_tokens untouched — no inaccurate estimate
    assert engine.state.usage.last_input_tokens == 150_000


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


@pytest.fixture
def llm_engine(creds_path: str) -> Generator[Engine]:
    """Engine wired with OpenAI credentials for LLM tests."""
    import queue

    from rbtr.creds import creds
    from rbtr.engine import Engine, EngineState
    from rbtr.sessions.store import SessionStore

    creds.update(openai_api_key="sk-test")
    state = EngineState(owner="testowner", repo_name="testrepo")
    state.openai_connected = True
    state.model_name = "openai/gpt-4o"
    eng = Engine(state, queue.Queue(), store=SessionStore())
    yield eng
    eng.close()


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
    from rbtr.engine.llm import _update_live_usage as _real_update

    mocker.patch(  # type: ignore[union-attr]
        "rbtr.engine.llm.build_model",
        return_value=_tool_then_text_model(),
    )
    summary_mock = mocker.patch(  # type: ignore[union-attr]
        "rbtr.engine.compact._stream_summary", return_value="Summary."
    )
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.engine.compact.build_model",
        return_value=_text_only_model(),
    )

    def _inflating_update(eng: Engine, run_usage: object, response: object) -> None:
        _real_update(eng, run_usage, response)  # type: ignore[arg-type]
        eng.state.usage.context_window = 50
        eng.state.usage.context_window_known = True

    mocker.patch("rbtr.engine.llm._update_live_usage", side_effect=_inflating_update)  # type: ignore[union-attr]
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
    assert any("compacting" in t.lower() for t in output_texts(events))


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
        "rbtr.engine.llm.build_model",
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
        "rbtr.engine.llm.build_model",
        return_value=_text_only_model(),
    )
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.engine.compact._stream_summary", return_value="Summary."
    )
    mocker.patch("rbtr.engine.compact.build_model")  # type: ignore[union-attr]
    llm_engine.state.usage.context_window = 50
    llm_engine.state.usage.context_window_known = True

    llm_engine.run_task("llm", "no tools")
    events = drain(llm_engine.events)
    texts = output_texts(events)

    assert not any("mid-turn" in t.lower() for t in texts)
