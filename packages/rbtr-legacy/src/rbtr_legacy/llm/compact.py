"""Context compaction — summarise older history to free context space."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai._agent_graph import ModelRequestNode
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from rbtr_legacy.config import ThinkingEffort, config
from rbtr_legacy.events import CompactionFinished, CompactionStarted
from rbtr_legacy.exceptions import RbtrError
from rbtr_legacy.prompts import render_compact, render_system
from rbtr_legacy.providers import build_model, model_settings
from rbtr_legacy.sessions.history import (
    build_summary_message,
    estimate_tokens,
    serialise_for_summary,
    snap_to_safe_boundary,
    split_history,
)
from rbtr_legacy.sessions.kinds import FragmentKind
from rbtr_legacy.sessions.overhead import (
    CompactionOverhead,
    CompactionTrigger,
    FactExtractionSource,
)

from .context import LLMContext
from .costs import extract_cost
from .memory import FactExtractionRun, apply_fact_extraction, run_fact_extraction

log = logging.getLogger(__name__)

# Minimum number of turns to keep — always preserve the most recent
# turn so the model has immediate context.
_MIN_KEEP_TURNS = 1

# Re-export for call sites that import from this module.
__all__ = ["CompactionTrigger", "compact_history", "compact_history_async", "reset_compaction"]


@dataclass(frozen=True, slots=True)
class _SummaryResult:
    """Internal result from `_stream_summary`."""

    text: str
    input_tokens: int
    output_tokens: int
    cost: float


# ── Compaction agent ─────────────────────────────────────────────────

compact_agent: Agent[None, str] = Agent()


@compact_agent.instructions
def _system() -> str:
    """Shared system prompt — same identity and language rules as the main agent."""
    return render_system()


@compact_agent.instructions
def _compact_task() -> str:
    """Compaction task — what to preserve and drop when summarising."""
    return render_compact()


# ── Public API ───────────────────────────────────────────────────────


def compact_history(
    ctx: LLMContext,
    extra_instructions: str = "",
    *,
    trigger: CompactionTrigger = CompactionTrigger.MANUAL,
) -> None:
    """Synchronous entry point — for daemon-thread callers.

    Schedules `compact_history_async` on the portal and blocks
    until it completes.  Safe to call from any thread **except** the
    portal's async task (that would deadlock).

    Used by `/compact` command and `_auto_compact_on_overflow`.
    """
    ctx.portal.call(lambda: compact_history_async(ctx, extra_instructions, trigger=trigger))


def reset_compaction(ctx: LLMContext) -> None:
    """Undo the latest compaction — restore compacted messages to active."""
    sid = ctx.state.session_id
    try:
        restored = ctx.store.reset_latest_compaction(sid)
    except ValueError as e:
        ctx.warn(str(e))
        return
    if restored == 0:
        ctx.out("Nothing to reset — session has no compacted messages.")
        return

    msgs = ctx.store.load_messages(sid)
    ctx.out(f"Compaction reset — {restored} fragments restored ({len(msgs)} active messages).")


async def compact_history_async(
    ctx: LLMContext,
    extra_instructions: str = "",
    *,
    trigger: CompactionTrigger = CompactionTrigger.MANUAL,
) -> None:
    """Summarise older messages to reduce context usage.

    Splits history into old and kept turns, serialises the old part,
    sends it to the current model for summarisation, and replaces
    history with `[summary] + kept`.

    When there are fewer turns than `keep_turns`, falls back to
    keeping only the last turn.  When the serialised old messages
    exceed the available context, only a prefix that fits is
    summarised — the rest is pushed into the kept portion.

    This is an async function so it can be `await`-ed from
    coroutines already running on the portal (mid-turn and post-turn
    compaction inside `_stream_agent`).
    """
    if not ctx.state.has_llm or not ctx.state.model_name:
        ctx.warn("No LLM connected — cannot compact.")
        return

    # Load history with DB row IDs — guarantees 1:1 alignment.
    sid = ctx.state.session_id
    paired = ctx.store.load_messages_with_ids(sid)
    history = [msg for _id, msg in paired]
    keep_turns = config.compaction.keep_turns
    old, kept = split_history(history, keep_turns)

    if not old and keep_turns > _MIN_KEEP_TURNS:
        old, kept = split_history(history, keep_turns=_MIN_KEEP_TURNS)

    if not old:
        ctx.out("Nothing to compact — conversation is short enough.")
        return

    # Collect DB row IDs for messages in `old`.  Repair may have
    # inserted synthetic messages (no DB ID) or replaced existing
    # ones, so match by object identity against the originals.
    old_set = {id(msg) for msg in old}
    old_ids = [mid for mid, msg in paired if id(msg) in old_set]

    max_tool_chars = config.compaction.summary_max_chars
    serialised = serialise_for_summary(old, max_tool_chars=max_tool_chars)

    reserve = config.compaction.reserve_tokens
    available = ctx.state.usage.context_window - reserve

    estimated = estimate_tokens(serialised)

    if estimated > available > 0:
        fit_count = find_fit_count(old, available, max_tool_chars)
        # Never split between a ModelResponse (tool calls) and
        # its ModelRequest (tool results).
        fit_count = snap_to_safe_boundary(old, fit_count)
        if fit_count == 0:
            ctx.warn("Cannot compact — even a single message exceeds available context.")
            return
        kept = list(old[fit_count:]) + kept
        old = list(old[:fit_count])
        old_set = {id(msg) for msg in old}
        old_ids = [mid for mid, msg in paired if id(msg) in old_set]
        serialised = serialise_for_summary(old, max_tool_chars=max_tool_chars)

    model_name = ctx.state.model_name
    if not model_name:
        ctx.warn("No model selected.")
        return

    try:
        model = build_model(model_name)
    except RbtrError as e:
        ctx.warn(str(e))
        return

    ctx.emit(CompactionStarted(old_messages=len(old), kept_messages=len(kept)))

    # Run summary and fact extraction concurrently — they are
    # independent (summary uses serialised text, fact extraction
    # uses raw messages) and hit separate agents.
    async def _extract_safe() -> FactExtractionRun | None:
        try:
            return await run_fact_extraction(
                old,
                ctx.store,
                ctx.state.repo_scope,
                model_name,
            )
        except Exception:
            log.exception("memory: fact extraction during compaction failed")
            return None

    summary_task = _stream_summary(ctx, model, extra_instructions, serialised)
    extract_task = _extract_safe()

    results = await asyncio.gather(summary_task, extract_task, return_exceptions=True)
    summary_result = results[0]
    extract_result = results[1]

    # Process fact extraction results and persist overhead (even if summary failed).
    if isinstance(extract_result, FactExtractionRun):
        await apply_fact_extraction(ctx, extract_result, FactExtractionSource.COMPACTION)

    if isinstance(summary_result, BaseException):
        ctx.error(f"Compaction failed: {summary_result}")
        ctx.emit(CompactionFinished(summary_tokens=0))
        return

    sr: _SummaryResult = summary_result
    summary_msg = build_summary_message(sr.text)

    ctx.store.compact_session(sid, summary=summary_msg, compact_ids=old_ids)
    ctx.state.usage.compaction_count += 1

    # Persist compaction overhead.
    summary_tokens = estimate_tokens(sr.text)
    ctx.store.save_overhead(
        sid,
        FragmentKind.OVERHEAD_COMPACTION,
        CompactionOverhead(
            trigger=trigger,
            old_messages=len(old),
            kept_messages=len(kept),
            summary_tokens=summary_tokens,
            model_name=ctx.state.model_name,
        ),
        input_tokens=sr.input_tokens,
        output_tokens=sr.output_tokens,
        cost=sr.cost,
    )
    ctx.state.usage.record_compaction(
        input_tokens=sr.input_tokens,
        output_tokens=sr.output_tokens,
        cost=sr.cost,
    )

    preview = sr.text[:200] + "…" if len(sr.text) > 200 else sr.text
    ctx.emit(CompactionFinished(summary_tokens=summary_tokens, summary_preview=preview))


def find_fit_count(
    messages: list[ModelMessage],
    available_tokens: int,
    max_tool_chars: int,
) -> int:
    """Find the largest prefix of *messages* whose serialised form fits.

    Uses binary search over message count.  Returns 0 when even a
    single message exceeds *available_tokens*.
    """
    lo, hi = 1, len(messages)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        text = serialise_for_summary(messages[:mid], max_tool_chars=max_tool_chars)
        if estimate_tokens(text) <= available_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


async def _stream_summary(
    ctx: LLMContext, model: Model, extra_instructions: str, conversation: str
) -> _SummaryResult:
    """Stream the compaction summary from the model.

    Returns a `_SummaryResult` with the summary text and overhead
    cost info from the LLM call.
    """
    effort = ThinkingEffort.NONE if ctx.state.effort_supported is False else config.thinking_effort
    settings = model_settings(ctx.state.model_name, model, effort)

    text_parts: list[str] = []

    async with compact_agent.iter(
        conversation,
        model=model,
        model_settings=settings,
        usage_limits=UsageLimits(request_limit=1),
        instructions=extra_instructions or None,
    ) as run:
        async for node in run:
            if isinstance(node, ModelRequestNode):
                async with node.stream(run.ctx) as stream:
                    async for delta in stream.stream_text(delta=True):
                        text_parts.append(delta)

    usage = run.usage()
    cost = extract_cost(run.new_messages())

    return _SummaryResult(
        text="".join(text_parts),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost=cost,
    )
