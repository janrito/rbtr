"""Handler for `/memory` — list, extract, and purge facts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rbtr_legacy.config import config
from rbtr_legacy.engine.session_cmd import parse_duration
from rbtr_legacy.llm.memory import extract_facts_from_ctx
from rbtr_legacy.sessions.kinds import GLOBAL_SCOPE

if TYPE_CHECKING:
    from .core import Engine


def cmd_memory(engine: Engine, args: str) -> None:
    """Dispatch `/memory` subcommands."""
    parts = args.split(maxsplit=1) if args else []
    sub = parts[0].lower() if parts else ""

    match sub:
        case "" | "list":
            _list_facts(engine, include_superseded=False)
        case "all":
            _list_facts(engine, include_superseded=True)
        case "extract":
            _extract(engine)
        case "purge":
            _purge(engine, parts[1] if len(parts) > 1 else "")
        case _:
            engine._warn("Usage: /memory [list | all | extract | purge <duration>]")


def _list_facts(engine: Engine, *, include_superseded: bool) -> None:
    """Print facts, optionally including superseded.

    Bare `/memory` shows global + current repo.
    `/memory all` shows every scope (all repos).
    """
    if not config.memory.enabled:
        engine._warn("Memory is disabled (config.memory.enabled = false).")
        return

    if include_superseded:
        # All scopes — discover from all facts, including superseded.
        all_scopes = engine.store.fact_scopes()
        if not all_scopes:
            engine._out("No facts stored yet.")
            return
        rest = sorted(s for s in all_scopes if s != GLOBAL_SCOPE)
        scopes = [GLOBAL_SCOPE, *rest] if GLOBAL_SCOPE in all_scopes else rest
    else:
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
        engine._out("")
        engine._markdown(f"### {label} ({len(facts)})")
        for f in facts:
            prefix = "~~" if f.superseded_by else ""
            suffix = "~~" if f.superseded_by else ""
            confirmed = f" (x{f.confirm_count})" if f.confirm_count > 1 else ""
            engine._markdown(f"- {prefix}{f.content}{suffix}{confirmed}")
        total += len(facts)

    if total == 0:
        engine._out("No facts stored yet.")
    else:
        engine._context(f"[/memory → {total} facts]", f"Listed {total} active facts.")


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
    engine._context("[/memory extract]", "Extracted facts from current session.")


def _purge(engine: Engine, args: str) -> None:
    """Delete facts older than a duration."""
    if not args:
        engine._warn("Usage: /memory purge <duration> (e.g. 7d, 2w, 24h)")
        return

    duration = parse_duration(args.strip())
    if duration is None:
        engine._warn(f"Invalid duration: {args.strip()}. Use e.g. 7d, 2w, 24h.")
        return

    cutoff = datetime.now(UTC) - duration
    deleted = engine.store.delete_old_facts(before=cutoff)
    engine._out(f"Deleted {deleted} fact{'s' if deleted != 1 else ''} older than {args.strip()}.")
