"""End-to-end invariant tests for the core behavioral guarantees.

1. History is immutable — DB content unchanged after transient repairs.
2. Failed tool calls don't break the conversation.
3. Compaction produces valid history.
4. Compaction works with failed-tool history.
5. Broken history is recovered transparently.
"""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pytest_cases import case, parametrize_with_cases

from rbtr_legacy.config import config
from rbtr_legacy.engine.core import Engine
from rbtr_legacy.engine.types import TaskType
from rbtr_legacy.events import TaskFinished
from rbtr_legacy.llm.compact import compact_agent, compact_history
from tests.engine.builders import _USAGE, _resp
from tests.helpers import StubProvider, drain
from tests.sessions.assertions import assert_ordering, assert_tool_pairing
from tests.sessions.case_histories import case_tool_failure

# ── 2. Failed tool call recovery ─────────────────────────────────────


@parametrize_with_cases("history", cases=[case_tool_failure])
def test_conversation_continues_after_tool_failure(
    history: list[ModelMessage], llm_engine: Engine, stub_provider: StubProvider
) -> None:
    """A tool failure (RetryPromptPart) doesn't break the conversation.

    Seed history with a failed tool call, send a follow-up,
    verify the model responds and history is valid.
    """
    llm_engine.store.save_messages(llm_engine.state.session_id, list(history))

    llm_engine.run_task(TaskType.LLM, "try config.py instead")
    events = drain(llm_engine.events)

    assert any(isinstance(e, TaskFinished) and e.success for e in events)

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert_ordering(loaded)
    assert "try config.py instead" in [
        p.content
        for m in loaded
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart) and isinstance(p.content, str)
    ]


# ── 3. Compaction produces valid history ─────────────────────────────


@parametrize_with_cases(
    "history",
    cases="tests.sessions.case_histories",
    has_tag="compactable",
)
def test_compaction_produces_valid_history(
    history: list[ModelMessage], engine: Engine, stub_provider: StubProvider
) -> None:
    """Compaction of every history shape produces valid output."""
    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    engine._sync_store_context()
    engine.store.save_messages(engine.state.session_id, list(history))
    engine.state.usage.context_window = 200_000
    config.memory.enabled = False

    ctx = engine._llm_context()
    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(ctx)
    drain(engine.events)

    loaded = engine.store.load_messages(engine.state.session_id)
    assert len(loaded) > 0
    assert_ordering(loaded)
    assert_tool_pairing(loaded)


# ── 4. Compaction + failed tool history ──────────────────────────────


@parametrize_with_cases("history", cases=[case_tool_failure])
def test_compaction_with_failed_tools_produces_valid_history(
    history: list[ModelMessage], engine: Engine, stub_provider: StubProvider
) -> None:
    """Compaction of history containing RetryPromptPart is valid."""

    engine.state.connected_providers.add("test")
    engine.state.model_name = "test/default"
    engine._sync_store_context()
    engine.store.save_messages(engine.state.session_id, history)
    engine.state.usage.context_window = 200_000
    config.memory.enabled = False

    ctx = engine._llm_context()
    with compact_agent.override(model=TestModel(custom_output_text="Summary.")):
        compact_history(ctx)
    drain(engine.events)

    loaded = engine.store.load_messages(engine.state.session_id)
    assert len(loaded) > 0
    assert len(loaded) < len(history)
    assert_ordering(loaded)
    assert_tool_pairing(loaded)


# ── 5. Broken history recovery ───────────────────────────────────────

# Intentionally malformed histories — each exercises a different
# level-0 repair in `_prepare_turn`.  Inline as `@case` functions
# discovered via `cases="."`.


@case(tags=["broken"])
def case_dangling_tool_call() -> list[ModelMessage]:
    """Tool call at end of history with no return — cancelled mid-tool."""
    return [
        ModelRequest(parts=[UserPromptPart(content="check files")]),
        _resp(
            TextPart(content="Checking."),
            ToolCallPart(tool_name="read_file", args={"path": "x.py"}, tool_call_id="tc1"),
        ),
        # Missing ToolReturnPart — simulates Ctrl+C.
    ]


@case(tags=["broken"])
def case_orphaned_tool_return() -> list[ModelMessage]:
    """Tool return with no matching call — can happen after compaction."""
    return [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="list_files", content="a.py", tool_call_id="gone"),
                UserPromptPart(content="what files are there?"),
            ]
        ),
        ModelResponse(
            parts=[TextPart(content="I see a.py.")],
            usage=_USAGE,
            model_name="test",
        ),
    ]


@case(tags=["broken"])
def case_corrupt_tool_args() -> list[ModelMessage]:
    """Corrupt tool-call args from a streaming failure."""
    return [
        ModelRequest(parts=[UserPromptPart(content="read it")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="read_file",
                    args='{"path": "a.py",\n<parameter name="offset": 10}',
                    tool_call_id="tc1",
                ),
            ],
            model_name="test",
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="contents", tool_call_id="tc1"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="Done.")], model_name="test"),
    ]


@case(tags=["broken"])
def case_gemini_tool_ids() -> list[ModelMessage]:
    """Gemini-style tool IDs with dots and colons."""
    return [
        ModelRequest(parts=[UserPromptPart(content="search")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="grep",
                    args={"pattern": "TODO"},
                    tool_call_id="functions.grep:4",
                ),
            ],
            model_name="gemini",
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="grep", content="found", tool_call_id="functions.grep:4"),
            ]
        ),
        ModelResponse(parts=[TextPart(content="Done.")], model_name="gemini"),
    ]


@parametrize_with_cases("history", cases=".", has_tag="broken")
def test_engine_recovers_from_broken_history(
    history: list[ModelMessage],
    llm_engine: Engine,
    stub_provider: StubProvider,
) -> None:
    """Level-0 repairs handle each broken shape transparently."""
    llm_engine.store.save_messages(llm_engine.state.session_id, list(history))

    llm_engine.run_task(TaskType.LLM, "continue")
    events = drain(llm_engine.events)

    assert any(isinstance(e, TaskFinished) and e.success for e in events)

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert_ordering(loaded)
