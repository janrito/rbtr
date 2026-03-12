"""Shared helpers for LLM tool modules.

Toolset instances, filter functions, accessor helpers, and output
formatting shared across tool modules.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import pygit2
from pydantic_ai import FunctionToolset, RunContext
from pydantic_ai.tools import ToolDefinition

from rbtr.index.store import IndexStore
from rbtr.llm.deps import AgentDeps
from rbtr.models import PRTarget

# ── Toolset instances ────────────────────────────────────────────────
#
# All four toolsets live here so tool modules (file.py, git.py, etc.)
# can register on the same instance without importing each other.

index_toolset: FunctionToolset[AgentDeps] = FunctionToolset()
"""Index tools — `search`, `changed_symbols`, `find_references`, `read_symbol`, `list_symbols`."""

repo_toolset: FunctionToolset[AgentDeps] = FunctionToolset()
"""Git + file tools — `changed_files`, `diff`, `commit_log`, `read_file`, `list_files`, `grep`."""

review_toolset: FunctionToolset[AgentDeps] = FunctionToolset()
"""PR feedback tools — `add_draft_comment`, `edit_draft_comment`, etc."""

workspace_toolset: FunctionToolset[AgentDeps] = FunctionToolset()
"""Persistent workspace tools — `edit`, `remember`."""


# ── Filter functions (used by FilteredToolset wrappers) ──────────────


def has_index(ctx: RunContext[AgentDeps], _tool_def: ToolDefinition) -> bool:
    """True when both the code index and a review target exist."""
    return ctx.deps.state.index is not None and ctx.deps.state.review_target is not None


def has_repo(ctx: RunContext[AgentDeps], _tool_def: ToolDefinition) -> bool:
    """True when both a git repository and a review target exist."""
    return ctx.deps.state.repo is not None and ctx.deps.state.review_target is not None


def has_pr_target(ctx: RunContext[AgentDeps], _tool_def: ToolDefinition) -> bool:
    """True when a PR target is selected (no GitHub auth needed)."""
    return isinstance(ctx.deps.state.review_target, PRTarget)


# ── Per-tool prepare functions (exceptions — only for stricter gates) ─


async def require_pr(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when a PR target and GitHub auth are available.

    Used as a per-tool prepare on `get_pr_discussion` — stricter
    than the group-level `has_pr_target` filter.
    """
    state = ctx.deps.state
    if state.gh is None or not isinstance(state.review_target, PRTarget):
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


# ── Toolset introspection ────────────────────────────────────────────


def _index_tool_names() -> list[str]:
    """Return names of tools registered on the index toolset."""
    return sorted(index_toolset.tools.keys())
