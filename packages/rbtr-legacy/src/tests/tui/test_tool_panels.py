"""Tests for concurrent tool-call panel rendering.

Covers `_truncate_head_tail`, `_render_head_tail`,
`_LivePanel` dataclass, and `tool_call_id` threading
through `_emit_tool_event`.
"""

from __future__ import annotations

import queue as _queue
from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from rich.text import Text

from rbtr_legacy.events import ToolCallFinished, ToolCallStarted
from rbtr_legacy.llm.stream import _emit_tool_event
from rbtr_legacy.tui.ui import (
    _LivePanel,
    _render_head_tail,
    _truncate_head_tail,
)

if TYPE_CHECKING:
    from rbtr_legacy.engine.core import Engine
    from rbtr_legacy.llm.context import LLMContext

pytest_plugins = ["tests.llm.conftest"]


# ── Helpers ──────────────────────────────────────────────────────────


def _lines(n: int) -> str:
    """Build a string with *n* numbered lines."""
    return "\n".join(f"line {i}" for i in range(1, n + 1))


def _drain(engine: Engine) -> list[ToolCallStarted | ToolCallFinished]:
    """Drain all tool-call events from the engine's event queue."""
    events: list[ToolCallStarted | ToolCallFinished] = []
    while True:
        try:
            ev = engine.events.get_nowait()
        except _queue.Empty:
            break
        if isinstance(ev, (ToolCallStarted, ToolCallFinished)):
            events.append(ev)
    return events


# ── _truncate_head_tail ─────────────────────────────────────────────


def test_truncate_short_text_no_truncation() -> None:
    head, hidden, tail = _truncate_head_tail(_lines(5), head_max=3, tail_max=12)
    assert head == _lines(5)
    assert hidden == 0
    assert tail == ""


def test_truncate_exact_budget_no_truncation() -> None:
    head, hidden, tail = _truncate_head_tail(_lines(15), head_max=3, tail_max=12)
    assert head == _lines(15)
    assert hidden == 0
    assert tail == ""


def test_truncate_one_over_budget() -> None:
    text = _lines(16)
    head, hidden, tail = _truncate_head_tail(text, head_max=3, tail_max=12)
    assert head == _lines(3)
    assert hidden == 1
    assert tail == "\n".join(text.splitlines()[-12:])


def test_truncate_large_text() -> None:
    text = _lines(100)
    head, hidden, tail = _truncate_head_tail(text, head_max=3, tail_max=12)
    assert head == _lines(3)
    assert hidden == 85
    assert tail == "\n".join(text.splitlines()[-12:])


def test_truncate_custom_budget() -> None:
    text = _lines(20)
    head, hidden, tail = _truncate_head_tail(text, head_max=5, tail_max=5)
    assert head == _lines(5)
    assert hidden == 10
    assert tail == "\n".join(text.splitlines()[-5:])


def test_truncate_empty_text() -> None:
    head, hidden, tail = _truncate_head_tail("", head_max=3, tail_max=12)
    assert head == ""
    assert hidden == 0
    assert tail == ""


def test_truncate_single_line() -> None:
    head, hidden, tail = _truncate_head_tail("hello", head_max=3, tail_max=12)
    assert head == "hello"
    assert hidden == 0
    assert tail == ""


# ── _render_head_tail ────────────────────────────────────────────────


def test_render_no_hidden_single_text() -> None:
    parts = _render_head_tail("content", 0, "", "dim")
    assert len(parts) == 1
    assert isinstance(parts[0], Text)
    assert isinstance(parts[0], Text)
    assert parts[0].plain == "content"


def test_render_empty_head_no_hidden() -> None:
    assert _render_head_tail("", 0, "", "dim") == []


def test_render_hidden_three_parts() -> None:
    parts = _render_head_tail("head", 5, "tail", "dim")
    assert len(parts) == 3
    assert isinstance(parts[0], Text)
    assert parts[0].plain == "head"
    assert isinstance(parts[1], Text)
    assert "5 more lines" in parts[1].plain
    assert isinstance(parts[2], Text)
    assert parts[2].plain == "tail"


def test_render_elapsed_in_spacer() -> None:
    parts = _render_head_tail("head", 5, "tail", "dim", elapsed=1.5)
    assert isinstance(parts[1], Text)
    assert "1.5s" in parts[1].plain


def test_render_no_elapsed_in_spacer() -> None:
    parts = _render_head_tail("head", 5, "tail", "dim")
    assert isinstance(parts[1], Text)
    assert "s)" not in parts[1].plain


def test_render_hidden_empty_tail() -> None:
    parts = _render_head_tail("head", 5, "", "dim")
    assert len(parts) == 2
    assert isinstance(parts[1], Text)
    assert "5 more lines" in parts[1].plain


def test_render_hidden_empty_head() -> None:
    parts = _render_head_tail("", 5, "tail", "dim")
    assert len(parts) == 2
    assert isinstance(parts[0], Text)
    assert "5 more lines" in parts[0].plain
    assert isinstance(parts[1], Text)
    assert parts[1].plain == "tail"


# ── _LivePanel ────────────────────────────────────────────────────


def test_live_panel_defaults() -> None:
    panel = _LivePanel(lines=[], variant="toolcall", tool_name="read_file")
    assert not panel.done
    assert panel.hidden == 0
    assert panel.expanded_lines is None
    assert panel.tool_name == "read_file"
    assert panel.tool_args == ""


def test_live_panel_state_tracking() -> None:
    panel = _LivePanel(lines=[], variant="toolcall")
    panel.done = True
    panel.variant = "failed"
    panel.hidden = 5
    panel.expanded_lines = [Text("full output")]
    assert panel.done
    assert panel.variant == "failed"
    assert panel.hidden == 5
    assert panel.expanded_lines is not None


# ── tool_call_id threading ───────────────────────────────────────────


def test_emit_tool_call_has_tool_call_id(engine: Engine, llm_ctx: LLMContext) -> None:
    """ToolCallStarted carries tool_call_id from the pydantic_ai event."""
    part = ToolCallPart(tool_name="read_file", args={"path": "x"}, tool_call_id="tc_42")
    _emit_tool_event(llm_ctx, FunctionToolCallEvent(part=part))

    events = _drain(engine)
    assert len(events) == 1
    assert isinstance(events[0], ToolCallStarted)
    assert events[0].tool_call_id == "tc_42"


def test_emit_tool_result_has_tool_call_id(engine: Engine, llm_ctx: LLMContext) -> None:
    """ToolCallFinished carries tool_call_id from the pydantic_ai event."""
    result = ToolReturnPart(tool_name="read_file", content="ok", tool_call_id="tc_99")
    _emit_tool_event(llm_ctx, FunctionToolResultEvent(result=result))

    events = _drain(engine)
    assert len(events) == 1
    assert isinstance(events[0], ToolCallFinished)
    assert events[0].tool_call_id == "tc_99"


def test_emit_tool_retry_has_tool_call_id(engine: Engine, llm_ctx: LLMContext) -> None:
    """ToolCallFinished from a retry carries tool_call_id."""
    result = RetryPromptPart(tool_name="edit", content="bad args", tool_call_id="tc_77")
    _emit_tool_event(llm_ctx, FunctionToolResultEvent(result=result))

    events = _drain(engine)
    assert len(events) == 1
    assert isinstance(events[0], ToolCallFinished)
    assert events[0].tool_call_id == "tc_77"
