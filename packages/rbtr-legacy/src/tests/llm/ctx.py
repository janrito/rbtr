"""RunContext builder for direct tool-function tests."""

from __future__ import annotations

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from rbtr.llm.deps import AgentDeps
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState


def build_tool_ctx(state: EngineState, store: SessionStore) -> RunContext[AgentDeps]:
    """Build a `RunContext[AgentDeps]` for direct tool-function calls."""
    return RunContext[AgentDeps](
        deps=AgentDeps(state=state, store=store),
        model=TestModel(),
        usage=RunUsage(),
    )
