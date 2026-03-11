"""Memory tool — let the agent save durable facts for future reviews."""

from __future__ import annotations

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from rbtr.config import config
from rbtr.llm.agent import AgentDeps, agent
from rbtr.llm.memory import GLOBAL_SCOPE


async def _require_memory(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Show the tool only when cross-session memory is enabled."""
    if not config.memory.enabled:
        return None
    return tool_def


@agent.tool(prepare=_require_memory)
def remember(
    ctx: RunContext[AgentDeps],
    content: str,
    scope: str = "global",
    supersedes: str | None = None,
) -> str:
    """Save a durable fact for future reviews.

    Use this to record project conventions, architecture decisions,
    user preferences, or recurring patterns discovered during review.
    Facts persist across sessions and are automatically injected
    into future conversations.

    Args:
        content: The fact to remember — a single, clear statement.
        scope: ``"global"`` for facts that apply everywhere, or
            ``"repo"`` for facts specific to the current repository.
        supersedes: The exact text of an existing fact to replace.
            Use when an older fact is outdated or incorrect.
    """
    store = ctx.deps.store
    state = ctx.deps.state
    session_id = state.session_id

    # Resolve scope.
    if scope == "repo":
        repo_scope = state.repo_scope
        if repo_scope is None:
            return "Cannot save a repo-scoped fact — no repository is connected."
        resolved_scope = repo_scope
    elif scope == "global":
        resolved_scope = GLOBAL_SCOPE
    else:
        return f"Invalid scope `{scope}` — use `global` or `repo`."

    # Handle supersession.
    old_fact = None
    if supersedes:
        old_fact = store.find_fact_by_content(supersedes.strip(), resolved_scope)
        if old_fact is None:
            return (
                f"No active fact matching that text in scope `{resolved_scope}`. "
                "Check the exact wording and scope."
            )

    # Insert the new fact.
    fact = store.insert_fact(resolved_scope, content, session_id)

    # Mark old fact as superseded.
    if old_fact:
        store.supersede_fact(old_fact.id, fact.id)

    parts = [f"Saved ({resolved_scope})."]
    if old_fact:
        parts.append("Superseded old fact.")
    return " ".join(parts)
