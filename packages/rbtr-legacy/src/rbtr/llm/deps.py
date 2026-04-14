"""Agent dependencies — shared type injected into every agent run."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rbtr.events import Event

if TYPE_CHECKING:
    from rbtr.sessions.store import SessionStore
    from rbtr.state import EngineState


@dataclass
class AgentDeps:
    """Dependencies injected into every agent run."""

    state: EngineState
    store: SessionStore
    events: queue.Queue[Event] = field(default_factory=queue.Queue)
    """Event queue for streaming tool output to the TUI."""
    cancel: threading.Event = field(default_factory=threading.Event)
    """Cancellation signal shared with the engine."""
