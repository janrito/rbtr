"""Agent dependencies — shared type injected into every agent run."""

from __future__ import annotations

from dataclasses import dataclass

from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState


@dataclass
class AgentDeps:
    """Dependencies injected into every agent run."""

    state: EngineState
    store: SessionStore
