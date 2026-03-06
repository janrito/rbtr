"""Shared helpers for LLM tool modules.

Prepare functions control tool visibility based on engine state.
Accessor helpers extract typed values from ``RunContext``.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import pygit2
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from rbtr.index.store import IndexStore
from rbtr.llm.agent import AgentDeps, agent
from rbtr.models import PRTarget

# ── Prepare functions ────────────────────────────────────────────────


async def require_index(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when an index is available."""
    if ctx.deps.state.index is None or ctx.deps.state.review_target is None:
        return None
    return tool_def


async def require_repo(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when a repo + review target is available."""
    if ctx.deps.state.repo is None or ctx.deps.state.review_target is None:
        return None
    return tool_def


async def require_pr(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when a PR target and GitHub auth are available."""
    state = ctx.deps.state
    if state.gh is None or not isinstance(state.review_target, PRTarget):
        return None
    return tool_def


async def require_pr_target(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when a PR target is selected.

    Unlike ``require_pr``, does not require GitHub auth — draft
    management is purely local.
    """
    if not isinstance(ctx.deps.state.review_target, PRTarget):
        return None
    return tool_def


# ── Accessor helpers ─────────────────────────────────────────────────


def head_commit(ctx: RunContext[AgentDeps]) -> str:
    """Return the git-resolvable head commit from the review target."""
    target = ctx.deps.state.review_target
    if target is None:  # pragma: no cover — guarded by prepare
        msg = "no review target"
        raise RuntimeError(msg)
    return target.head_commit


def get_store(ctx: RunContext[AgentDeps]) -> IndexStore:
    """Return the index store."""
    store = ctx.deps.state.index
    if store is None:  # pragma: no cover — guarded by prepare
        msg = "no index store"
        raise RuntimeError(msg)
    return store


def get_repo(ctx: RunContext[AgentDeps]) -> pygit2.Repository:
    """Return the git repo."""
    repo = ctx.deps.state.repo
    if repo is None:  # pragma: no cover — guarded by prepare
        msg = "no repository"
        raise RuntimeError(msg)
    return repo


def resolve_tool_ref(ctx: RunContext[AgentDeps], ref: str) -> str:
    """Map `"head"` / `"base"` to the review target's branch names.

    Any other value is returned as-is (raw git ref).
    """
    target = ctx.deps.state.review_target
    match ref:
        case "head":
            if target is None:  # pragma: no cover — guarded by prepare
                msg = "no review target"
                raise RuntimeError(msg)
            return target.head_commit
        case "base":
            if target is None:  # pragma: no cover — guarded by prepare
                msg = "no review target"
                raise RuntimeError(msg)
            return target.base_commit
        case _:
            return ref


# ── Output limiting ──────────────────────────────────────────────────


def validate_path(path: str) -> str | None:
    """Return an error message if *path* is invalid, else ``None``."""
    if ".." in PurePosixPath(path).parts:
        return f"Path '{path}' contains '..' — must be relative to repo root."
    return None


def limited(shown: int, total: int, *, hint: str) -> str:
    """Standard truncation trailer appended when output is capped.

    Every tool that caps output uses this, so the LLM sees a
    consistent format and knows how to request more.
    """
    return f"\n\n... limited ({shown}/{total}). {hint}"


# ── Agent introspection ──────────────────────────────────────────────


def _index_tool_names() -> list[str]:
    """Return names of tools that use :func:`require_index` as their prepare function."""
    return sorted(
        name
        for name, tool in agent._function_toolset.tools.items()
        if tool.prepare is require_index
    )
