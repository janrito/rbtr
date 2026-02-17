"""Agent definition — pydantic-ai Agent with decorator-based configuration.

The agent is defined once at module level.  Instructions use the
``@agent.instructions`` decorator so they receive ``RunContext`` and
can read live session state.

The model is provided at each call site via ``agent.iter(model=...)``,
not baked into the agent.  Future tools and output validation plug in
via ``@agent.tool`` and ``output_type`` on the same instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from rbtr.engine.session import Session
from rbtr.prompts import render_review, render_system


@dataclass
class AgentDeps:
    """Dependencies injected into every agent run."""

    session: Session


agent: Agent[AgentDeps, str] = Agent(deps_type=AgentDeps)


@agent.instructions
def system_prompt(ctx: RunContext[AgentDeps]) -> str:
    """Render the main system prompt with live session context."""
    return render_system(ctx.deps.session)


@agent.instructions
def review_guidelines(ctx: RunContext[AgentDeps]) -> str:
    """Render the static review guidelines."""
    return render_review()
