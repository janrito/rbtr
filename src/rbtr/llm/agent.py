"""Agent definition — pydantic-ai Agent with decorator-based configuration.

The agent is defined once at module level.  Instructions use
``@agent.instructions`` decorators — stateless ones (system) are
plain functions, stateful ones (review, index status) receive
``RunContext``.

The model is provided at each call site via ``agent.iter(model=...)``,
not baked into the agent.  Tools plug in via ``@agent.tool`` on the
same instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from rbtr.prompts import render_index_status, render_review, render_system
from rbtr.state import EngineState


@dataclass
class AgentDeps:
    """Dependencies injected into every agent run."""

    state: EngineState


agent: Agent[AgentDeps, str] = Agent(deps_type=AgentDeps)


@agent.instructions
def _system() -> str:
    """Shared system prompt — identity, language, project rules."""
    return render_system()


@agent.instructions
def _review_task(ctx: RunContext[AgentDeps]) -> str:
    """Review task — context, principles, strategy, format."""
    return render_review(ctx.deps.state)


# Import tools so @agent.tool decorators execute and register.
import rbtr.llm.tools  # noqa: E402, F401
from rbtr.llm.tools.common import _index_tool_names  # noqa: E402


@agent.instructions
def _index_status(ctx: RunContext[AgentDeps]) -> str:
    """Render index status instruction from the template."""
    state = ctx.deps.state
    if state.review_target is None:
        return ""
    status = "ready" if state.index_ready else "building"
    return render_index_status(status=status, tool_names=_index_tool_names())
