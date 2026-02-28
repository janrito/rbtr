"""Context compaction — summarise older history to free context space."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_ai._agent_graph import ModelRequestNode
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from rbtr.config import config
from rbtr.events import CompactionFinished, CompactionStarted
from rbtr.exceptions import RbtrError
from rbtr.prompts import render_compact
from rbtr.providers import build_model

from .history import (
    build_summary_message,
    estimate_tokens,
    serialise_for_summary,
    snap_to_safe_boundary,
    split_history,
)

if TYPE_CHECKING:
    from .core import Engine

# Minimum number of turns to keep — always preserve the most recent
# turn so the model has immediate context.
_MIN_KEEP_TURNS = 1


def compact_history(engine: Engine, extra_instructions: str = "") -> None:
    """Synchronous entry point — for daemon-thread callers.

    Schedules :func:`compact_history_async` on the engine's event loop
    and blocks until it completes.  Safe to call from any thread
    **except** the event-loop thread itself (that would deadlock).

    Used by ``/compact`` command and ``_auto_compact_on_overflow``.
    """
    future = asyncio.run_coroutine_threadsafe(
        compact_history_async(engine, extra_instructions),
        engine._loop,
    )
    future.result()


def reset_compaction(engine: Engine) -> None:
    """Undo the latest compaction — restore compacted messages to active."""
    sid = engine.state.session_id
    try:
        restored = engine.store.reset_latest_compaction(sid)
    except ValueError as e:
        engine._warn(str(e))
        return
    if restored == 0:
        engine._out("Nothing to reset — session has no compacted messages.")
        return

    msgs = engine.store.load_messages(sid)
    engine._out(f"Compaction reset — {restored} fragments restored ({len(msgs)} active messages).")


async def compact_history_async(engine: Engine, extra_instructions: str = "") -> None:
    """Summarise older messages to reduce context usage.

    Splits history into old and kept turns, serialises the old part,
    sends it to the current model for summarisation, and replaces
    history with ``[summary] + kept``.

    When there are fewer turns than ``keep_turns``, falls back to
    keeping only the last turn.  When the serialised old messages
    exceed the available context, only a prefix that fits is
    summarised — the rest is pushed into the kept portion.

    This is an async function so it can be ``await``-ed from
    coroutines already running on ``engine._loop`` (mid-turn and
    post-turn compaction inside ``_stream_agent``).
    """
    if not engine.state.has_llm:
        engine._warn("No LLM connected — cannot compact.")
        return

    # Load history with DB row IDs — guarantees 1:1 alignment.
    sid = engine.state.session_id
    paired = engine.store.load_messages_with_ids(sid)
    history = [msg for _id, msg in paired]
    keep_turns = config.compaction.keep_turns
    old, kept = split_history(history, keep_turns)

    if not old and keep_turns > _MIN_KEEP_TURNS:
        old, kept = split_history(history, keep_turns=_MIN_KEEP_TURNS)

    if not old:
        engine._out("Nothing to compact — conversation is short enough.")
        return

    # Collect IDs for old messages.  split_history may move orphaned
    # tool returns from kept into old, so positional slicing is wrong —
    # match by object identity instead.
    old_set = {id(msg) for msg in old}
    old_ids = [mid for mid, msg in paired if id(msg) in old_set]

    max_tool_chars = config.compaction.summary_max_chars
    serialised = serialise_for_summary(old, max_tool_chars=max_tool_chars)

    reserve = config.compaction.reserve_tokens
    available = engine.state.usage.context_window - reserve
    estimated = estimate_tokens(serialised)

    if estimated > available > 0:
        fit_count = find_fit_count(old, available, max_tool_chars)
        # Never split between a ModelResponse (tool calls) and
        # its ModelRequest (tool results).
        fit_count = snap_to_safe_boundary(old, fit_count)
        if fit_count == 0:
            engine._warn("Cannot compact — even a single message exceeds available context.")
            return
        kept = list(old[fit_count:]) + kept
        old = list(old[:fit_count])
        old_set = {id(msg) for msg in old}
        old_ids = [mid for mid, msg in paired if id(msg) in old_set]
        serialised = serialise_for_summary(old, max_tool_chars=max_tool_chars)

    try:
        model = build_model(engine.state.model_name)
    except RbtrError as e:
        engine._warn(str(e))
        return

    engine._emit(CompactionStarted(old_messages=len(old), kept_messages=len(kept)))

    instructions = render_compact(extra_instructions)

    try:
        summary_text = await _stream_summary(engine, model, instructions, serialised)
    except (RbtrError, ModelHTTPError, OSError) as e:
        engine._error(f"Compaction failed: {e}")
        engine._emit(CompactionFinished(summary_tokens=0))
        return

    summary_msg = build_summary_message(summary_text)

    engine.store.compact_session(sid, summary=summary_msg, compact_ids=old_ids)
    engine.state.usage.compaction_count += 1

    engine._emit(CompactionFinished(summary_tokens=estimate_tokens(summary_text)))


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
    engine: Engine, model: Model, instructions: str, conversation: str
) -> str:
    """Stream the summary from the model, return the full text."""
    from .llm import resolve_model_settings  # deferred: avoid circular at module level

    settings = resolve_model_settings(
        model, engine.state.model_name, effort_supported=engine.state.effort_supported
    )

    summary_agent: Agent[None, str] = Agent(model, instructions=instructions)
    text_parts: list[str] = []

    async with summary_agent.iter(
        conversation,
        model=model,
        model_settings=settings,
        usage_limits=UsageLimits(request_limit=1),
    ) as run:
        async for node in run:
            if isinstance(node, ModelRequestNode):
                async with node.stream(run.ctx) as stream:
                    async for delta in stream.stream_text(delta=True):
                        text_parts.append(delta)

    return "".join(text_parts)
