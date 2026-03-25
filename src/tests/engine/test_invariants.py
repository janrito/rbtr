"""End-to-end invariant tests for the core behavioral guarantees.

1. History is immutable — DB content unchanged after transient repairs.
2. Failed tool calls don't break the conversation.
3. Compaction produces valid history.
4. Compaction works with failed-tool history.
5. Broken history is recovered transparently.
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from rbtr.config import config
from rbtr.engine.core import Engine
from rbtr.engine.types import TaskType
from rbtr.events import TaskFinished
from rbtr.llm.compact import compact_agent, compact_history
from tests.engine.test_compact import ALL_HISTORIES
from tests.helpers import TestProvider, drain
from tests.sessions.assertions import assert_ordering, assert_tool_pairing

_USAGE = RequestUsage(input_tokens=0, output_tokens=0)


def _resp(*parts: TextPart | ToolCallPart) -> ModelResponse:
    return ModelResponse(parts=list(parts), usage=_USAGE, model_name="test")


# ── 1. History immutability ──────────────────────────────────────────


def test_repair_does_not_modify_db(llm_engine: Engine, test_provider: TestProvider) -> None:
    """Transient repairs don't alter persisted history.

    Seed a dangling tool call (no return), run an LLM turn,
    verify the DB still has the dangling call — repairs are in-memory only.
    """
    dangling: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="read my files")]),
        ModelResponse(
            parts=[
                TextPart(content="Let me check."),
                ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
            ],
            model_name="test",
        ),
        # No ToolReturnPart — simulates Ctrl+C mid-tool-call.
    ]
    sid = llm_engine.state.session_id
    llm_engine.store.save_messages(sid, dangling)

    # Snapshot DB before the LLM turn.
    before = llm_engine.store.load_messages(sid)

    # Run a turn — _prepare_turn applies transient repair.
    llm_engine.run_task(TaskType.LLM, "continue")
    drain(llm_engine.events)

    # The original 2 messages are unchanged in DB.
    after_raw = llm_engine.store.load_messages(sid)
    # First 2 messages should be the originals (dangling).
    assert len(after_raw) >= len(before)
    for orig, loaded in zip(before, after_raw[: len(before)], strict=True):
        assert type(orig) is type(loaded)
        assert len(orig.parts) == len(loaded.parts)


# ── 2. Failed tool call recovery ─────────────────────────────────────


def test_conversation_continues_after_tool_failure(
    llm_engine: Engine, test_provider: TestProvider
) -> None:
    """A tool failure (RetryPromptPart) doesn't break the conversation.

    Seed history with a failed tool call, send a follow-up,
    verify the model responds and history is valid.
    """
    history_with_failure: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="read secret.py")]),
        _resp(
            TextPart(content="Reading."),
            ToolCallPart(tool_name="read_file", args={"path": "secret.py"}, tool_call_id="c1"),
        ),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    content="Permission denied",
                    tool_name="read_file",
                    tool_call_id="c1",
                ),
            ]
        ),
        ModelResponse(
            parts=[TextPart(content="Can't read that file.")],
            usage=_USAGE,
            model_name="test",
        ),
    ]
    llm_engine.store.save_messages(llm_engine.state.session_id, history_with_failure)

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


@pytest.mark.parametrize(
    "history_name",
    [
        "text_only",
        "single_tool",
        "parallel_tools",
        "chained_tools",
        "tool_failure",
        "thinking_with_tools",
        "reordered_returns",
        "tool_no_preamble",
    ],
)
def test_compaction_produces_valid_history(
    history_name: str, engine: Engine, test_provider: TestProvider
) -> None:
    """Compaction of every history shape produces valid output."""

    history = ALL_HISTORIES[history_name]
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


def test_compaction_with_failed_tools_produces_valid_history(
    engine: Engine, test_provider: TestProvider
) -> None:
    """Compaction of history containing RetryPromptPart is valid."""

    # 4 turns with a tool failure in turn 1.
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="read secret.py")]),
        _resp(
            TextPart(content="Reading."),
            ToolCallPart(tool_name="read_file", args={"path": "secret.py"}, tool_call_id="c1"),
        ),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    content="Permission denied",
                    tool_name="read_file",
                    tool_call_id="c1",
                ),
            ]
        ),
        ModelResponse(
            parts=[TextPart(content="Can't read that.")],
            usage=_USAGE,
            model_name="test",
        ),
        ModelRequest(parts=[UserPromptPart(content="try config.py")]),
        _resp(
            ToolCallPart(tool_name="read_file", args={"path": "config.py"}, tool_call_id="c2"),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="DEBUG=True", tool_call_id="c2"),
            ]
        ),
        ModelResponse(
            parts=[TextPart(content="config.py has DEBUG=True.")],
            usage=_USAGE,
            model_name="test",
        ),
        ModelRequest(parts=[UserPromptPart(content="any other issues?")]),
        ModelResponse(
            parts=[TextPart(content="No issues.")],
            usage=_USAGE,
            model_name="test",
        ),
        ModelRequest(parts=[UserPromptPart(content="ship it")]),
        ModelResponse(
            parts=[TextPart(content="LGTM.")],
            usage=_USAGE,
            model_name="test",
        ),
    ]

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


def test_engine_recovers_from_dangling_tool_call(
    llm_engine: Engine, test_provider: TestProvider
) -> None:
    """Engine handles history with a dangling tool call — no crash.

    Seed a dangling call (response with ToolCallPart, no ToolReturnPart),
    send a message, verify the task succeeds.
    """
    dangling: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="check files")]),
        _resp(
            TextPart(content="Checking."),
            ToolCallPart(tool_name="read_file", args={"path": "x.py"}, tool_call_id="tc1"),
        ),
        # Missing ToolReturnPart — cancelled mid-tool.
    ]
    llm_engine.store.save_messages(llm_engine.state.session_id, dangling)

    llm_engine.run_task(TaskType.LLM, "continue from where you left off")
    events = drain(llm_engine.events)

    assert any(isinstance(e, TaskFinished) and e.success for e in events)

    loaded = llm_engine.store.load_messages(llm_engine.state.session_id)
    assert len(loaded) >= 4  # original 2 + new request + response


def test_engine_recovers_from_orphaned_tool_return(
    llm_engine: Engine, test_provider: TestProvider
) -> None:
    """Engine handles history with an orphaned tool return — no crash.

    Seed a return with no matching call (can happen after compaction
    of straddling tool calls), send a message, verify success.
    """
    orphaned: list[ModelMessage] = [
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
    llm_engine.store.save_messages(llm_engine.state.session_id, orphaned)

    llm_engine.run_task(TaskType.LLM, "anything else?")
    events = drain(llm_engine.events)

    assert any(isinstance(e, TaskFinished) and e.success for e in events)
