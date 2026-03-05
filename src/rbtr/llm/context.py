"""LLMContext — everything the LLM pipeline needs, no Engine dependency."""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass

from rbtr.events import ErrorDetail, Event, Output
from rbtr.exceptions import TaskCancelled
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState
from rbtr.styles import STYLE_DIM, STYLE_ERROR, STYLE_WARNING


@dataclass
class LLMContext:
    """Dependencies for the LLM streaming pipeline.

    Created by the engine before each LLM or compaction call.
    Provides event emission, session persistence, and cancellation
    without coupling to the ``Engine`` class.
    """

    state: EngineState
    store: SessionStore
    events: queue.Queue[Event]
    cancel: threading.Event
    loop: asyncio.AbstractEventLoop

    def emit(self, event: Event) -> None:
        """Put an event on the queue."""
        self.events.put(event)

    def out(self, text: str) -> None:
        """Emit an informational message.  Checks cancellation first."""
        if self.cancel.is_set():
            raise TaskCancelled
        self.events.put(Output(text=text, style=STYLE_DIM))

    def warn(self, text: str) -> None:
        """Emit a warning.  Checks cancellation first."""
        if self.cancel.is_set():
            raise TaskCancelled
        self.events.put(Output(text=text, style=STYLE_WARNING))

    def error(self, text: str) -> None:
        """Emit an error.  Always emitted — no cancellation check."""
        self.events.put(Output(text=text, style=STYLE_ERROR))

    def error_with_detail(self, summary: str, detail: str) -> None:
        """Emit an error with expandable diagnostic detail."""
        if self.cancel.is_set():
            raise TaskCancelled
        self.events.put(ErrorDetail(summary=summary, detail=detail))
