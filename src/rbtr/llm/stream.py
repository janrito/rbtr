"""Handler for LLM queries — streaming via the shared agent."""

from __future__ import annotations

import asyncio
import collections.abc
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    HandleResponseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits

from rbtr.config import ThinkingEffort, config
from rbtr.events import TextDelta, ToolCallFinished, ToolCallStarted
from rbtr.exceptions import RbtrError, TaskCancelled
from rbtr.providers import build_model, model_context_window

from .agent import AgentDeps, agent
from .compact import compact_history, compact_history_async
from .context import LLMContext
from .errors import is_context_overflow, is_effort_unsupported
from .history import (
    demote_thinking,
    flatten_tool_exchanges,
    is_history_format_error,
    repair_dangling_tool_calls,
)
from .model_settings import resolve_model_settings

if TYPE_CHECKING:
    from rbtr.sessions.store import ResponseWriter

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _StreamResult:
    """Result of a single agent streaming run."""

    all_messages: list[ModelMessage]
    new_messages: list[ModelMessage]
    usage: RunUsage
    limit_hit: bool
    last_writer: ResponseWriter | None
    compact_needed: bool = False


_LIMIT_SUMMARY_PROMPT = (
    "You have reached the tool-call limit for this turn. "
    "Summarize what you accomplished so far and what remains to be done, "
    "so the user can decide whether to ask you to continue."
)


def handle_llm(ctx: LLMContext, message: str) -> None:
    """Send a message to the active LLM, streaming the response."""
    if not ctx.state.has_llm:
        ctx.warn("No LLM connected. Use /connect claude, chatgpt, or openai.")
        return

    try:
        model = build_model(ctx.state.model_name)
    except RbtrError as e:
        ctx.warn(str(e))
        return

    try:
        _run_agent(ctx, model, message)
    except ModelHTTPError as exc:
        if exc.status_code == 400 and is_effort_unsupported(exc):
            ctx.state.effort_supported = False
            ctx.out("Model does not support effort — retrying without it…")
            _run_agent(ctx, model, message)
            return
        if exc.status_code == 400 and is_history_format_error(exc):
            ctx.out("Retrying with simplified history…")
            _run_agent(ctx, model, message, simplify_history=True)
            return
        if is_context_overflow(exc):
            _auto_compact_on_overflow(ctx, message)
            return
        raise
    except ValueError as exc:
        if _is_tool_args_error(exc):
            ctx.out("Retrying with simplified history…")
            _run_agent(ctx, model, message, simplify_history=True)
            return
        raise


def _is_tool_args_error(exc: ValueError) -> bool:
    """Check if a ValueError is from malformed tool-call args.

    Provider adapters call ``ToolCallPart.args_as_dict()`` which
    uses ``pydantic_core.from_json``.  If the model produced
    invalid JSON for tool arguments during streaming (e.g. mixed
    XML/JSON), the error surfaces here as a ``ValueError``.

    Normally ``_validate_tool_call_args`` in the deserialisation
    layer catches these at load time.  This handler is a fallback
    for edge cases (e.g. args corrupted after loading).
    """
    msg = str(exc).lower()
    return "key must be a string" in msg or "eof while parsing" in msg


def _auto_compact_on_overflow(ctx: LLMContext, message: str) -> None:
    """Compact history and retry after a context-overflow error.

    ``compact_history`` handles all edge cases internally (no LLM,
    too few messages, LLM failure).  If compaction doesn't help,
    the retry raises the same overflow and the caller handles it.
    """
    ctx.out("Context limit reached — compacting history…")
    compact_history(ctx)
    ctx.out("Retrying with compacted history…")
    handle_llm(ctx, message)


# ── Agent run ────────────────────────────────────────────────────────


def _finish_response_writer(writer: ResponseWriter, all_msgs: list[ModelMessage]) -> None:
    """Close a response writer with per-request token counts.

    Extracts usage from the last ``ModelResponse`` (the one just
    streamed) for resilience — the final cost is added later by
    ``_finalize_turn``.
    """
    last = all_msgs[-1] if all_msgs else None
    if isinstance(last, ModelResponse):
        writer.finish(
            input_tokens=last.usage.input_tokens,
            output_tokens=last.usage.output_tokens,
            cache_read_tokens=last.usage.cache_read_tokens or None,
            cache_write_tokens=last.usage.cache_write_tokens or None,
        )
    else:
        writer.finish()


def _needs_mid_turn_compaction(ctx: LLMContext, response: ModelResponse) -> bool:
    """Check whether context usage warrants mid-turn compaction.

    Only triggers when the response contains tool calls (meaning more
    requests will follow).  Text-only responses end the turn naturally.
    """
    has_tool_calls = any(isinstance(p, ToolCallPart) for p in response.parts)
    return has_tool_calls and ctx.state.usage.context_used_pct >= config.compaction.auto_compact_pct


async def _do_stream(
    ctx: LLMContext,
    model: Model,
    deps: AgentDeps,
    settings: ModelSettings | None,
    prompt: str | None,
    msg_history: list[ModelMessage],
) -> _StreamResult:
    """Execute one ``agent.iter()`` pass — stream model output and tool calls.

    This is the single streaming loop shared by the main turn, mid-turn
    compaction resume, and the limit-hit summary.
    """
    store = ctx.store
    saved_count = len(msg_history)
    saved_request_count = 0
    limit_hit = False
    compact_needed = False
    last_writer: ResponseWriter | None = None

    async with agent.iter(
        prompt,
        model=model,
        deps=deps,
        message_history=msg_history or None,
        model_settings=settings,
        usage_limits=UsageLimits(request_limit=config.tools.max_requests_per_turn),
    ) as run:
        try:
            async for node in run:
                match node:
                    # ── Model response: stream text deltas, persist parts ──
                    case ModelRequestNode():
                        writer = store.begin_response(
                            ctx.state.session_id,
                            model_name=ctx.state.model_name,
                        )
                        last_writer = writer

                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                match event:
                                    case PartStartEvent(index=idx, part=part):
                                        writer.add_part(idx, part)
                                    case PartDeltaEvent(delta=delta):
                                        if isinstance(delta, TextPartDelta):
                                            ctx.emit(TextDelta(delta=delta.content_delta))
                                    case PartEndEvent(index=idx, part=part):
                                        writer.finish_part(idx, part)

                        _finish_response_writer(writer, run.all_messages())

                    # ── Tool calls: execute tools, check compaction ──
                    case CallToolsNode():
                        _update_live_usage(ctx, run.usage(), node.model_response)

                        async with node.stream(run.ctx) as tool_stream:
                            async for tool_event in tool_stream:
                                _emit_tool_event(ctx, tool_event)

                        if _needs_mid_turn_compaction(ctx, node.model_response):
                            compact_needed = True

                # ── Persist after every node ──
                all_msgs = run.all_messages()
                n = _save_new_requests(ctx, all_msgs, saved_count)
                saved_request_count += n
                saved_count = len(all_msgs)

                if compact_needed:
                    break
        except UsageLimitExceeded:
            limit_hit = True

    new = list(run.new_messages())
    requests: list[ModelMessage] = [m for m in new if isinstance(m, ModelRequest)]
    if len(requests) > saved_request_count:
        _save_messages_safe(ctx, requests[saved_request_count:])

    return _StreamResult(
        all_messages=list(run.all_messages()),
        new_messages=new,
        usage=run.usage(),
        limit_hit=limit_hit,
        last_writer=last_writer,
        compact_needed=compact_needed,
    )


def _run_agent(
    ctx: LLMContext,
    model: Model,
    message: str,
    *,
    simplify_history: bool = False,
) -> None:
    """Stream an agent run, blocking until complete."""
    future = asyncio.run_coroutine_threadsafe(
        _stream_agent(ctx, model, message, simplify_history=simplify_history),
        ctx.loop,
    )
    future.result()


def _prepare_turn(
    ctx: LLMContext,
    model: Model,
    simplify_history: bool,
) -> tuple[list[ModelMessage], AgentDeps, ModelSettings | None]:
    """Load history, repair dangling tool calls, resolve settings.

    Mutates engine state: emits warnings for repaired tool calls,
    persists synthetic repair messages, tracks ``effort_supported``,
    and snapshots the usage baseline for the upcoming run.
    """
    store = ctx.store
    history = store.load_messages(ctx.state.session_id)
    history, repaired_tools, repair_messages = repair_dangling_tool_calls(history)
    if repaired_tools:
        _save_messages_safe(ctx, repair_messages)
        names = ", ".join(repaired_tools)
        ctx.warn(
            f"Previous turn was cancelled mid-tool-call ({names}). "
            f"Those tool results are lost — the model will continue without them."
        )
    if simplify_history:
        history = demote_thinking(history)
        history = flatten_tool_exchanges(history)

    deps = AgentDeps(state=ctx.state)
    settings = resolve_model_settings(
        model, ctx.state.model_name, effort_supported=ctx.state.effort_supported
    )
    if (
        config.thinking_effort is not ThinkingEffort.NONE
        and ctx.state.effort_supported is not False
    ):
        ctx.state.effort_supported = settings is not None

    ctx.state.usage.snapshot_base()
    return history, deps, settings


def _finalize_turn(ctx: LLMContext, result: _StreamResult) -> None:
    """Record usage and write cost to the last response writer.

    The writer's per-request tokens were already set during streaming
    (for resilience if the run crashes).  This adds cost — the only
    value that requires the full run to compute.
    """
    run_cost, cost_available = _record_usage(ctx, result.new_messages, result.usage)
    if result.last_writer is not None:
        last_resp = next(
            (m for m in reversed(result.new_messages) if isinstance(m, ModelResponse)),
            None,
        )
        result.last_writer.finish(
            cost=run_cost if cost_available else None,
            input_tokens=last_resp.usage.input_tokens if last_resp else None,
            output_tokens=last_resp.usage.output_tokens if last_resp else None,
            cache_read_tokens=(last_resp.usage.cache_read_tokens or None) if last_resp else None,
            cache_write_tokens=(last_resp.usage.cache_write_tokens or None) if last_resp else None,
        )


async def _stream_agent(
    ctx: LLMContext,
    model: Model,
    message: str,
    *,
    simplify_history: bool = False,
) -> None:
    """Run the agent with cancellation support, update session state."""
    store = ctx.store
    history, deps, settings = _prepare_turn(ctx, model, simplify_history)

    result = await _run_with_cancel(ctx, _do_stream(ctx, model, deps, settings, message, history))

    # Mid-turn compaction: if context exceeded the threshold during
    # a tool-call cycle, compact once and resume.
    if result.compact_needed:
        await compact_history_async(
            ctx,
            extra_instructions="The model is mid-turn with active tool calls.",
        )
        # Reload from DB after compaction.
        history = store.load_messages(ctx.state.session_id)
        ctx.state.usage.snapshot_base()

        resume = await _run_with_cancel(ctx, _do_stream(ctx, model, deps, settings, None, history))
        result = _merge_results(result, resume)

    _finalize_turn(ctx, result)

    if result.limit_hit:
        history = store.load_messages(ctx.state.session_id)
        summary = await _run_with_cancel(
            ctx, _do_stream(ctx, model, deps, settings, _LIMIT_SUMMARY_PROMPT, history)
        )
        _save_messages_safe(ctx, [m for m in summary.new_messages if isinstance(m, ModelRequest)])

    # Post-turn compaction for turns that didn't trigger mid-turn.
    if (
        not result.compact_needed
        and ctx.state.usage.context_used_pct >= config.compaction.auto_compact_pct
    ):
        await compact_history_async(ctx)


# ── Persistence helpers ──────────────────────────────────────────────


def _save_new_requests(
    ctx: LLMContext,
    all_messages: list[ModelMessage],
    saved_count: int,
) -> int:
    """Save new ``ModelRequest`` messages that appeared since *saved_count*.

    Responses are persisted incrementally via ``begin_response``.
    Returns the number of request messages saved.
    """
    new = all_messages[saved_count:]
    requests: list[ModelMessage] = [m for m in new if isinstance(m, ModelRequest)]
    if requests:
        _save_messages_safe(ctx, requests)
    return len(requests)


def _save_messages_safe(ctx: LLMContext, messages: list[ModelMessage]) -> None:
    """Save messages to the store, logging failures instead of raising.

    Relies on ``engine._sync_store_context()`` having been called at
    task start — metadata is inherited from the stored context.
    """
    if not messages:
        return
    try:
        ctx.store.save_messages(ctx.state.session_id, messages)
    except OSError:
        log.warning("sessions: failed to persist messages", exc_info=True)


async def _run_with_cancel(
    ctx: LLMContext,
    coro: collections.abc.Coroutine[object, object, _StreamResult],
) -> _StreamResult:
    """Run *coro* with a parallel cancellation watcher.

    Raises ``TaskCancelled`` if the engine's cancel event fires
    before the coroutine completes.
    """

    async def _watch() -> None:
        while not ctx.cancel.is_set():
            await asyncio.sleep(0.1)
        raise TaskCancelled

    stream_task = asyncio.create_task(coro)
    cancel_task = asyncio.create_task(_watch())

    done, pending = await asyncio.wait(
        {stream_task, cancel_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    for task in done:
        if task is cancel_task:
            task.result()  # raises TaskCancelled

    return stream_task.result()


def _merge_results(first: _StreamResult, second: _StreamResult) -> _StreamResult:
    """Combine two sequential stream results (pre-compact + post-compact)."""
    return _StreamResult(
        all_messages=second.all_messages,
        new_messages=first.new_messages + second.new_messages,
        usage=_merge_usage(first.usage, second.usage),
        limit_hit=first.limit_hit or second.limit_hit,
        last_writer=second.last_writer or first.last_writer,
        compact_needed=False,
    )


def _merge_usage(a: RunUsage, b: RunUsage) -> RunUsage:
    """Sum two ``RunUsage`` instances."""
    return RunUsage(
        requests=a.requests + b.requests,
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
    )


# ── Event helpers ────────────────────────────────────────────────────


def _emit_tool_event(
    ctx: LLMContext,
    event: HandleResponseEvent,
) -> None:
    """Emit a ToolCallStarted or ToolCallFinished event."""
    match event:
        case FunctionToolCallEvent(part=part):
            args = part.args
            if isinstance(args, dict):
                args_str = json.dumps(args, ensure_ascii=False)
            elif args is not None:
                args_str = str(args)
            else:
                args_str = ""
            ctx.emit(ToolCallStarted(tool_name=part.tool_name, args=args_str))
        case FunctionToolResultEvent(result=result):
            if isinstance(result, ToolReturnPart):
                text = str(result.content)
                max_chars = config.tui.tool_max_chars
                if len(text) > max_chars:
                    text = text[:max_chars] + "…"
            else:
                text = "(retry)"
            tool_name = getattr(result, "tool_name", None) or "?"
            ctx.emit(ToolCallFinished(tool_name=tool_name, result=text))


# ── Usage tracking ───────────────────────────────────────────────────


def _update_live_usage(
    ctx: LLMContext,
    run_usage: RunUsage,
    response: ModelResponse,
) -> None:
    """Snapshot mid-run usage into the session for live footer display.

    Called after each model request completes (before tool execution).
    Uses cumulative ``run_usage`` for totals, and the per-request
    ``response.usage.input_tokens`` for context-% so the footer
    updates progressively during a multi-tool-call turn.

    Does **not** increment ``response_count`` or cost — the
    authoritative ``_record_usage`` at run-end handles those.
    """
    usage = ctx.state.usage
    base = usage.live_base
    usage.input_tokens = base.input_tokens + run_usage.input_tokens
    usage.output_tokens = base.output_tokens + run_usage.output_tokens
    usage.cache_read_tokens = base.cache_read_tokens + run_usage.cache_read_tokens
    usage.cache_write_tokens = base.cache_write_tokens + run_usage.cache_write_tokens
    usage.last_input_tokens = response.usage.input_tokens

    # Set context window from model metadata so the footer shows the
    # correct value during streaming — works for both endpoints and
    # built-in providers (Anthropic, OpenAI, etc.).
    _apply_model_context_window(ctx)


def _apply_model_context_window(ctx: LLMContext) -> None:
    """Set the context window from model metadata if available.

    No-op when the context window is already known.  Checks endpoint
    metadata first, then falls back to genai-prices for built-in
    providers.
    """
    if ctx.state.usage.context_window_known:
        return
    cw = model_context_window(ctx.state.model_name)
    if cw is not None:
        ctx.state.usage.context_window = cw
        ctx.state.usage.context_window_known = True


def _record_usage(
    ctx: LLMContext,
    new_messages: list[ModelMessage],
    run_usage: RunUsage,
) -> tuple[float, bool]:
    """Extract cost/context-window from new messages and update session usage.

    Returns ``(run_cost, cost_available)`` so callers can persist cost.

    ``run_usage.input_tokens`` is the **sum** across all requests in
    the run (PydanticAI accumulates).  For context-% we need the
    *last* request's input tokens -- that's the actual prompt size the
    model received, reflecting the full conversation + tool results.
    We pull it from the last ``ModelResponse.usage.input_tokens``.
    """
    run_cost = 0.0
    cost_available = False
    context_window: int | None = None
    last_input_tokens: int | None = None

    for msg in new_messages:
        if not isinstance(msg, ModelResponse) or not msg.model_name:
            continue
        # Each ModelResponse carries per-request usage -- the last one
        # is the most recent prompt size (what we display as context %).
        last_input_tokens = msg.usage.input_tokens
        try:
            price = msg.cost()
            run_cost += float(price.total_price)
            cost_available = True
            if price.model and price.model.context_window:
                context_window = price.model.context_window
        except (AssertionError, LookupError, ValueError):
            pass

    # Prefer model metadata (endpoint config or genai-prices lookup)
    # over the value from msg.cost(), which may not know the model.
    meta_context_window = model_context_window(ctx.state.model_name)
    if meta_context_window is not None:
        context_window = meta_context_window

    response_count = sum(1 for m in new_messages if isinstance(m, ModelResponse))
    ctx.state.usage.record_run(
        input_tokens=run_usage.input_tokens,
        output_tokens=run_usage.output_tokens,
        cache_read_tokens=run_usage.cache_read_tokens,
        cache_write_tokens=run_usage.cache_write_tokens,
        last_input_tokens=last_input_tokens,
        cost=run_cost,
        cost_available=cost_available,
        context_window=context_window,
        new_responses=response_count,
    )

    return run_cost, cost_available
