"""LLMContext — everything the LLM pipeline needs, no Engine dependency."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anyio.from_thread import BlockingPortal
from anyio.lowlevel import EventLoopToken

from rbtr_legacy.events import Event, Output, OutputLevel
from rbtr_legacy.exceptions import TaskCancelled
from rbtr_legacy.llm.types import CancelSlot

if TYPE_CHECKING:
    from rbtr_legacy.sessions.store import SessionStore
    from rbtr_legacy.state import EngineState


@dataclass
class LLMContext:
    """Dependencies for the LLM streaming pipeline.

    Created by the engine before each LLM or compaction call.
    Provides event emission, session persistence, and cancellation
    without coupling to the `Engine` class.
    """

    state: EngineState
    store: SessionStore
    events: queue.Queue[Event]
    cancel: threading.Event
    portal: BlockingPortal
    anyio_token: EventLoopToken
    cancel_slot: CancelSlot = field(default_factory=lambda: [None])

    def emit(self, event: Event) -> None:
        """Put an event on the queue."""
        self.events.put(event)

    def out(self, text: str) -> None:
        """Emit an informational message.  Checks cancellation first."""
        if self.cancel.is_set():
            raise TaskCancelled
        self.events.put(Output(text=text))

    def warn(self, text: str) -> None:
        """Emit a warning.  Checks cancellation first."""
        if self.cancel.is_set():
            raise TaskCancelled
        self.events.put(Output(text=text, level=OutputLevel.WARNING))

    def error(self, text: str) -> None:
        """Emit an error.  Always emitted — no cancellation check."""
        self.events.put(Output(text=text, level=OutputLevel.ERROR))

    def error_with_detail(self, summary: str, detail: str) -> None:
        """Emit an error with expandable diagnostic detail."""
        if self.cancel.is_set():
            raise TaskCancelled
        self.events.put(Output(text=summary, level=OutputLevel.ERROR, detail=detail))
