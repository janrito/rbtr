"""End-to-end replay roundtrip tests.

Drives the full engine pipeline, captures live events, replays
stored history, compares.  Scenarios are ``@case`` functions.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel
from pytest_cases import parametrize_with_cases

from rbtr_legacy.config import config
from rbtr_legacy.engine.core import Engine
from rbtr_legacy.engine.types import TaskType
from rbtr_legacy.events import (
    FlushPanel,
    MarkdownOutput,
    Output,
    PanelVariant,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from rbtr_legacy.llm.compact import compact_agent, compact_history
from rbtr_legacy.sessions.history import replay_history
from tests.helpers import StubProvider, drain
from tests.sessions.case_scenarios import Scenario


def _texts(events: list[object]) -> list[str]:
    """Extract text content from events (TextDelta or MarkdownOutput)."""
    result: list[str] = []
    acc = ""
    for e in events:
        if isinstance(e, TextDelta):
            acc += e.delta
        elif isinstance(e, MarkdownOutput):
            acc += e.text
        elif acc:
            result.append(acc)
            acc = ""
    if acc:
        result.append(acc)
    return result


def _tool_ids(events: list[object]) -> set[str]:
    """Extract tool_call_ids from ToolCallStarted events."""
    return {e.tool_call_id for e in events if isinstance(e, ToolCallStarted)}


@parametrize_with_cases("scenario", cases="tests.sessions.case_scenarios")
def test_live_roundtrip(
    scenario: Scenario, llm_engine: Engine, stub_provider: StubProvider
) -> None:
    """Run scenario, replay, compare live and replay content."""
    config.memory.enabled = False

    live_events: list[object] = []
    for i, (model, prompt) in enumerate(scenario.turns):
        stub_provider.set_model(model)
        llm_engine.run_task(TaskType.LLM, prompt)
        live_events.extend(drain(llm_engine.events))

        if scenario.compact_after is not None and i + 1 == scenario.compact_after:
            llm_engine.state.usage.context_window = 200_000
            ctx = llm_engine._llm_context()
            with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
                compact_history(ctx)
            drain(llm_engine.events)

    replay_events: list[object] = []
    stored = llm_engine.store.load_messages(llm_engine.state.session_id)
    replay_history(replay_events.append, stored)

    # Text: replay content is a subset of live content.
    live_text = _texts(live_events)
    replay_text = _texts(replay_events)
    for rt in replay_text:
        assert any(rt in lt for lt in live_text), f"Replay text not in live: '{rt[:80]}'"

    # Tools: replay tool calls are a subset of live tool calls.
    assert _tool_ids(replay_events) <= _tool_ids(live_events)

    # Every ToolCallStarted has a matching ToolCallFinished.
    for e in replay_events:
        if isinstance(e, ToolCallStarted):
            assert any(
                isinstance(f, ToolCallFinished) and f.tool_call_id == e.tool_call_id
                for f in replay_events
            )

    # Compaction marker when expected.
    if scenario.expect_compaction_marker:
        assert any(isinstance(e, Output) for e in replay_events)
        assert any(
            isinstance(e, FlushPanel) and e.variant == PanelVariant.QUEUED for e in replay_events
        )


@parametrize_with_cases("messages", cases="tests.sessions.case_histories", has_tag="failure")
def test_failure_roundtrip(messages: list[ModelMessage], llm_engine: Engine) -> None:
    """Failure histories round-trip with errors preserved."""
    llm_engine._sync_store_context()
    llm_engine.store.save_messages(llm_engine.state.session_id, list(messages))

    replay_events: list[object] = []
    stored = llm_engine.store.load_messages(llm_engine.state.session_id)
    replay_history(replay_events.append, stored)

    assert any(isinstance(e, ToolCallStarted) for e in replay_events)
    assert any(isinstance(e, ToolCallFinished) and e.error for e in replay_events), (
        "Expected ToolCallFinished with error"
    )


@parametrize_with_cases("messages", cases="tests.sessions.case_histories")
def test_stored_history_replays_cleanly(messages: list[ModelMessage], llm_engine: Engine) -> None:
    """Every case shape survives save → load → replay."""
    llm_engine._sync_store_context()
    llm_engine.store.save_messages(llm_engine.state.session_id, list(messages))

    stored = llm_engine.store.load_messages(llm_engine.state.session_id)
    replay_events: list[object] = []
    replay_history(replay_events.append, stored)

    assert replay_events, "replay_history produced no events"

    # Every MarkdownOutput/Output is followed by FlushPanel.
    for i, event in enumerate(replay_events):
        if isinstance(event, (MarkdownOutput, Output)):
            assert i + 1 < len(replay_events), "output at end without FlushPanel"
            assert isinstance(replay_events[i + 1], FlushPanel)

    # Every ToolCallStarted has a matching ToolCallFinished.
    for event in replay_events:
        if isinstance(event, ToolCallStarted):
            assert any(
                isinstance(f, ToolCallFinished) and f.tool_call_id == event.tool_call_id
                for f in replay_events
            )
