"""Agent definition — pydantic-ai Agent with toolset-based tool registration.

The agent is created lazily on first use via `get_agent()`.
Tool submodules are registered inside `get_agent()` before
the agent is constructed, deferring `duckdb`/`pyarrow` until
the first LLM task.

The model is provided at each call site via `agent.iter(model=...)`,
not baked into the agent.

Tools are organised into `FunctionToolset` instances, each wrapped
in a `FilteredToolset` that gates the entire group on engine state.
Presentation order to the model follows toolset order x registration
order within each toolset.
"""

from __future__ import annotations

import functools

from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FilteredToolset

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


def register_tools() -> None:
    """Import tool submodules to trigger ``@toolset.tool`` registration.

    Defers `duckdb` + `pyarrow` (pulled in by `tools/index.py` →
    `orchestrator.py` → `index/store.py`) until the first LLM task.
    Import order determines tool presentation order within
    cross-module toolsets.
    """
    # isort: off
    from rbtr.llm.tools import index  # noqa: F401
    from rbtr.llm.tools import git  # noqa: F401
    from rbtr.llm.tools import file  # noqa: F401
    from rbtr.llm.tools import discussion  # noqa: F401
    from rbtr.llm.tools import draft  # noqa: F401
    from rbtr.llm.tools import notes  # noqa: F401
    from rbtr.llm.tools import memory  # noqa: F401
    from rbtr.llm.tools import shell  # noqa: F401
    # isort: on


@functools.cache
def get_agent() -> Agent[AgentDeps, str]:
    """Return the shared agent, creating it on first call.

    Registers all tool submodules and builds the agent with
    instructions. Heavy deps (`duckdb`/`pyarrow`) are loaded
    here, not at import time.
    """
    register_tools()

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
    def _system() -> str:
        return render_system()

    @agent.instructions
    def _review_task(ctx: RunContext[AgentDeps]) -> str:
        return render_review(ctx.deps.state)

    @agent.instructions
    def _index_status(ctx: RunContext[AgentDeps]) -> str:
        state = ctx.deps.state
        if state.review_target is None:
            return ""
        status = "ready" if state.index_ready else "building"
        return render_index_status(status=status, tool_names=_index_tool_names())

    @agent.instructions
    def _memory(ctx: RunContext[AgentDeps]) -> str:
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
        registry = ctx.deps.state.skill_registry
        if registry is None:
            return ""
        skills = registry.visible()
        if not skills:
            return ""
        return render_skills(skills)

    return agent
