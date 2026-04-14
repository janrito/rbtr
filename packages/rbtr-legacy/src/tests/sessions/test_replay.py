"""Tests for `rbtr.sessions.history.replay_history`.

Parametrized over the shared conversation cases so that every
history shape is tested.  Uses the same event types the live
streaming path produces — single rendering route.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage
from pytest_cases import parametrize_with_cases

from rbtr.events import (
    FlushPanel,
    InputEcho,
    MarkdownOutput,
    Output,
    PanelVariant,
    ToolCallFinished,
    ToolCallStarted,
)
from rbtr.sessions.history import replay_history

# ── Structural invariants (all cases) ────────────────────────────────


@parametrize_with_cases("messages", cases="tests.sessions.case_histories")
def test_replay_produces_events(messages: list[ModelMessage]) -> None:
    """Replay of every history shape produces at least one event."""
    events: list[object] = []
    replay_history(events.append, messages)
    assert events, "replay_history produced no events"


@parametrize_with_cases("messages", cases="tests.sessions.case_histories")
def test_replay_flushes_all_panels(messages: list[ModelMessage]) -> None:
    """Every MarkdownOutput or Output is followed by a FlushPanel."""
    events: list[object] = []
    replay_history(events.append, messages)
    for i, event in enumerate(events):
        if isinstance(event, (MarkdownOutput, Output)):
            assert i + 1 < len(events), "MarkdownOutput/Output at end without FlushPanel"
            assert isinstance(events[i + 1], FlushPanel), (
                f"Expected FlushPanel after {type(event).__name__}, got {type(events[i + 1]).__name__}"
            )


@parametrize_with_cases("messages", cases="tests.sessions.case_histories")
def test_replay_tool_calls_paired(messages: list[ModelMessage]) -> None:
    """Every ToolCallStarted is immediately followed by ToolCallFinished."""
    events: list[object] = []
    replay_history(events.append, messages)
    for i, event in enumerate(events):
        if isinstance(event, ToolCallStarted):
            assert i + 1 < len(events), "ToolCallStarted at end without ToolCallFinished"
            nxt = events[i + 1]
            assert isinstance(nxt, ToolCallFinished), (
                f"Expected ToolCallFinished after ToolCallStarted, got {type(nxt).__name__}"
            )
            assert nxt.tool_call_id == event.tool_call_id


@parametrize_with_cases("messages", cases="tests.sessions.case_histories")
def test_replay_response_variant(messages: list[ModelMessage]) -> None:
    """FlushPanel after MarkdownOutput uses the RESPONSE variant."""
    events: list[object] = []
    replay_history(events.append, messages)
    for i, event in enumerate(events):
        if isinstance(event, MarkdownOutput) and i + 1 < len(events):
            flush = events[i + 1]
            if isinstance(flush, FlushPanel):
                assert flush.variant == PanelVariant.RESPONSE


# ── Content-specific tests ───────────────────────────────────────────


@parametrize_with_cases("messages", cases="tests.sessions.case_histories", has_tag="tool")
def test_replay_emits_tool_events(messages: list[ModelMessage]) -> None:
    """Tool-containing histories produce ToolCallStarted events."""
    events: list[object] = []
    replay_history(events.append, messages)
    tool_starts = [e for e in events if isinstance(e, ToolCallStarted)]
    assert tool_starts, "Expected ToolCallStarted events for tool history"


@parametrize_with_cases("messages", cases="tests.sessions.case_histories", has_tag="failure")
def test_replay_shows_failed_tools(messages: list[ModelMessage]) -> None:
    """Histories with RetryPromptPart produce ToolCallFinished with error."""
    events: list[object] = []
    replay_history(events.append, messages)
    failed = [e for e in events if isinstance(e, ToolCallFinished) and e.error is not None]
    assert failed, "Expected at least one failed ToolCallFinished"


@parametrize_with_cases("messages", cases="tests.sessions.case_histories", has_tag="compaction")
def test_replay_compaction_summary(messages: list[ModelMessage]) -> None:
    """Compaction summary renders as Output in QUEUED panel, not InputEcho."""
    events: list[object] = []
    replay_history(events.append, messages)
    input_echos = [e for e in events if isinstance(e, InputEcho)]
    outputs = [e for e in events if isinstance(e, Output)]
    flushes = [e for e in events if isinstance(e, FlushPanel) and e.variant == PanelVariant.QUEUED]
    assert outputs, "Expected Output for compaction summary"
    assert flushes, "Expected FlushPanel(QUEUED) for compaction summary"
    for echo in input_echos:
        assert "compacted" not in echo.text.lower(), (
            "Compaction summary should not appear as InputEcho"
        )


@parametrize_with_cases("messages", cases="tests.sessions.case_histories", has_tag="thinking")
def test_replay_skips_thinking(messages: list[ModelMessage]) -> None:
    """ThinkingParts do not produce any events."""
    events: list[object] = []
    replay_history(events.append, messages)
    texts = [e.text for e in events if isinstance(e, (MarkdownOutput, Output, InputEcho))]
    combined = " ".join(texts)
    assert "Let me think" not in combined, "ThinkingPart content leaked into replay"
    assert "Simple entry point" not in combined, "ThinkingPart content leaked into replay"
