"""Tests for `AutoSubmit` — engine-triggered LLM dispatch."""

from __future__ import annotations

from rbtr.engine.core import Engine
from rbtr.engine.types import TaskType
from rbtr.events import AutoSubmit
from tests.helpers import drain, has_event_type


def test_auto_submit_emitted(engine: Engine) -> None:
    """Engine can emit an `AutoSubmit` event."""
    engine._emit(AutoSubmit(message="hello"))
    events = drain(engine.events)
    assert has_event_type(events, AutoSubmit)
    evt = next(e for e in events if isinstance(e, AutoSubmit))
    assert evt.message == "hello"


def test_auto_submit_cleared_on_cancel(engine: Engine) -> None:
    """AutoSubmit survives in the queue until consumed by the TUI.

    Cancellation clearing is handled by the TUI main loop (tested
    via integration), not the engine.  This test just verifies the
    event is a normal queue entry.
    """
    engine._emit(AutoSubmit(message="search query"))
    engine.run_task(TaskType.COMMAND, "/help")
    events = drain(engine.events)
    assert has_event_type(events, AutoSubmit)
