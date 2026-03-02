"""Agent definition — pydantic-ai Agent with decorator-based configuration.

The agent is defined once at module level.  System prompts use the
``@agent.system_prompt`` decorator so they receive ``RunContext`` and
can read live state.

The model is provided at each call site via ``agent.iter(model=...)``,
not baked into the agent.  Future tools and output validation plug in
via ``@agent.tool`` and ``output_type`` on the same instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from rbtr.prompts import render_index_status, render_system
from rbtr.state import EngineState


@dataclass
class AgentDeps:
    """Dependencies injected into every agent run."""

    state: EngineState


agent: Agent[AgentDeps, str] = Agent(deps_type=AgentDeps)


@agent.instructions
def system_prompt(ctx: RunContext[AgentDeps]) -> str:
    """Render the main system prompt with live state context."""
    return render_system(ctx.deps.state)


def _index_tool_names() -> list[str]:
    """Return names of tools that require the index, via introspection."""
    from rbtr.llm.tools import _require_index  # deferred: circular at import time

    return sorted(
        name
        for name, tool in agent._function_toolset.tools.items()
        if tool.prepare is _require_index
    )


@agent.instructions
def index_status(ctx: RunContext[AgentDeps]) -> str:
    """Render index status instruction from the template."""
    state = ctx.deps.state
    if state.review_target is None:
        return ""
    status = "ready" if state.index_ready else "building"
    return render_index_status(status=status, tool_names=_index_tool_names())


# Import tools so @agent.tool decorators execute and register.
import rbtr.llm.tools as _tools  # noqa: E402, F401  # side-effect import for tool registration
