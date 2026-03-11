"""Handler for ``/memory`` — list and extract facts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.config import config
from rbtr.llm.memory import GLOBAL_SCOPE, extract_facts_from_ctx

if TYPE_CHECKING:
    from .core import Engine


def cmd_memory(engine: Engine, args: str) -> None:
    """Dispatch ``/memory`` subcommands."""
    sub = args.split(maxsplit=1)[0].lower() if args else ""

    match sub:
        case "" | "list":
            _list_facts(engine, include_superseded=False)
        case "all":
            _list_facts(engine, include_superseded=True)
        case "extract":
            _extract(engine)
        case _:
            engine._warn("Usage: /memory [list | all | extract]")


def _list_facts(engine: Engine, *, include_superseded: bool) -> None:
    """Print active facts, optionally including superseded."""
    if not config.memory.enabled:
        engine._warn("Memory is disabled (config.memory.enabled = false).")
        return

    scopes = [GLOBAL_SCOPE]
    repo_scope = engine.state.repo_scope
    if repo_scope:
        scopes.append(repo_scope)

    total = 0
    for scope in scopes:
        if include_superseded:
            facts = engine.store.load_all_facts(scope)
        else:
            facts = engine.store.load_active_facts(scope)

        if not facts:
            continue

        label = "global" if scope == GLOBAL_SCOPE else scope
        engine._out(f"**{label}** ({len(facts)})")
        for f in facts:
            prefix = "~~" if f.superseded_by else ""
            suffix = "~~" if f.superseded_by else ""
            confirmed = f" (x{f.confirm_count})" if f.confirm_count > 1 else ""
            engine._out(f"  {prefix}{f.content}{suffix}{confirmed}")
        total += len(facts)

    if total == 0:
        engine._out("No facts stored yet.")


def _extract(engine: Engine) -> None:
    """Extract facts from the current session's active messages."""
    if not config.memory.enabled:
        engine._warn("Memory is disabled (config.memory.enabled = false).")
        return

    ctx = engine._llm_context()
    if not ctx.state.has_llm or not ctx.state.model_name:
        engine._warn("No LLM connected — cannot extract facts.")
        return

    messages = engine.store.load_messages(ctx.state.session_id)
    if not messages:
        engine._out("No messages in the current session.")
        return

    extract_facts_from_ctx(ctx, messages)
