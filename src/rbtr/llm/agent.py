"""Agent definition — pydantic-ai Agent with toolset-based tool registration.

The agent is defined once at module level.  Instructions use
`@agent.instructions` decorators — stateless ones (system) are
plain functions, stateful ones (review, index status, memory)
receive `RunContext`.

The model is provided at each call site via `agent.iter(model=...)`,
not baked into the agent.

Tools are organised into `FunctionToolset` instances, each wrapped
in a `FilteredToolset` that gates the entire group on engine state.
Presentation order to the model follows toolset order x registration
order within each toolset.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FilteredToolset

import rbtr.llm.tools.shell as _shell_tools  # noqa: F401  # registers run_command on shell_toolset
from rbtr.config import config
from rbtr.llm.deps import AgentDeps
from rbtr.llm.memory import render_facts_instruction
from rbtr.llm.tools.common import (
    _index_tool_names,
    diff_toolset,
    file_toolset,
    has_diff_target,
    has_index,
    has_pr_target,
    has_repo,
    has_shell,
    index_toolset,
    review_toolset,
    shell_toolset,
    workspace_toolset,
)
from rbtr.prompts import render_index_status, render_review, render_skills, render_system
from rbtr.providers import system_instructions

agent: Agent[AgentDeps, str] = Agent(
    deps_type=AgentDeps,
    toolsets=[
        FilteredToolset(index_toolset, filter_func=has_index),
        FilteredToolset(file_toolset, filter_func=has_repo),
        FilteredToolset(diff_toolset, filter_func=has_diff_target),
        FilteredToolset(review_toolset, filter_func=has_pr_target),
        FilteredToolset(shell_toolset, filter_func=has_shell),
        workspace_toolset,
    ],
)


@agent.instructions
def _provider_identity(ctx: RunContext[AgentDeps]) -> str:
    """Provider-specific system text, prepended before everything else."""
    return system_instructions(ctx.deps.state.model_name) or ""


@agent.instructions
def _system() -> str:
    """Shared system prompt — identity, language, project rules."""
    return render_system()


@agent.instructions
def _review_task(ctx: RunContext[AgentDeps]) -> str:
    """Review task — context, principles, strategy, format."""
    return render_review(ctx.deps.state)


@agent.instructions
def _index_status(ctx: RunContext[AgentDeps]) -> str:
    """Render index status instruction from the template."""
    state = ctx.deps.state
    if state.review_target is None:
        return ""
    status = "ready" if state.index_ready else "building"
    return render_index_status(status=status, tool_names=_index_tool_names())


@agent.instructions
def _memory(ctx: RunContext[AgentDeps]) -> str:
    """Inject facts from cross-session memory."""
    if not config.memory.enabled:
        return ""
    return render_facts_instruction(
        store=ctx.deps.store,
        repo_scope=ctx.deps.state.repo_scope,
        max_facts=config.memory.max_injected_facts,
        max_tokens=config.memory.max_injected_tokens,
    )


@agent.instructions
def _skills(ctx: RunContext[AgentDeps]) -> str:
    """Inject the available skills catalog."""
    registry = ctx.deps.state.skill_registry
    if registry is None:
        return ""
    skills = registry.visible()
    if not skills:
        return ""
    return render_skills(skills)
