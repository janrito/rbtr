"""Handler for LLM queries — streaming via the shared agent."""

from __future__ import annotations

import asyncio
import collections.abc
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent
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
from rbtr.exceptions import RbtrError
from rbtr.providers import (
    build_model,
    build_model_settings,
    endpoint_model_settings,
    model_context_window,
)

from .agent import AgentDeps, agent
from .history import demote_thinking, is_history_format_error
from .types import TaskCancelled

if TYPE_CHECKING:
    from rbtr.sessions.store import ResponseWriter

    from .core import Engine

log = logging.getLogger(__name__)


# ── History repair ───────────────────────────────────────────────────


def _repair_dangling_tool_calls(
    history: list[ModelMessage],
) -> tuple[list[ModelMessage], list[str]]:
    """Fix history left dirty by a cancelled tool-calling turn.

    When the user cancels (Ctrl+C) mid-turn, the model's
    ``ModelResponse`` with tool calls may already be persisted but
    the matching tool results are not.  This leaves the history in
    a broken state — both PydanticAI's own validation and upstream
    provider APIs (OpenAI, Anthropic) reject conversations that
    contain function calls without matching results.

    The fix: for every ``ModelResponse`` with ``ToolCallPart``\\ s
    that is *not* followed by a ``ModelRequest`` containing the
    corresponding ``ToolReturnPart``\\ s, append a synthetic request
    with ``(cancelled)`` results.

    Returns ``(repaired_history, tool_names)`` where *tool_names*
    lists every tool that was patched (empty if no repair needed).
    """
    if not history:
        return history, []

    repaired: list[ModelMessage] | None = None  # lazy copy
    all_tool_names: list[str] = []

    i = 0
    while i < len(history):
        msg = history[i]
        if isinstance(msg, ModelResponse):
            tool_calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
            if tool_calls:
                # Check whether the next message supplies results.
                next_msg = history[i + 1] if i + 1 < len(history) else None
                has_results = isinstance(next_msg, ModelRequest) and any(
                    isinstance(p, ToolReturnPart) for p in next_msg.parts
                )
                if not has_results:
                    if repaired is None:
                        repaired = list(history[:i])
                    repaired.append(msg)
                    names = [tc.tool_name for tc in tool_calls]
                    all_tool_names.extend(names)
                    log.info(
                        "Repairing %d dangling tool call(s) from cancelled turn.",
                        len(tool_calls),
                    )
                    repaired.append(
                        ModelRequest(
                            parts=[
                                ToolReturnPart(
                                    tool_name=tc.tool_name,
                                    content="(cancelled)",
                                    tool_call_id=tc.tool_call_id,
                                )
                                for tc in tool_calls
                            ],
                        )
                    )
                    i += 1
                    continue
        if repaired is not None:
            repaired.append(msg)
        i += 1

    if repaired is not None:
        return repaired, all_tool_names
    return history, []


@dataclass(slots=True)
class _StreamResult:
    """Result of a single agent streaming run."""

    all_messages: list[ModelMessage]
    new_messages: list[ModelMessage]
    usage: RunUsage
    limit_hit: bool
    last_writer: ResponseWriter | None
    compact_needed: bool = False


def resolve_model_settings(
    model: Model,
    model_name: str | None,
    *,
    effort_supported: bool | None = None,
) -> ModelSettings | None:
    """Build merged model settings (thinking effort + endpoint overrides).

    Used by both the main agent run and compaction summaries to avoid
    duplicating settings-construction logic.

    When *effort_supported* is ``False``, the effort parameter is
    omitted even if ``config.thinking_effort`` is set — this avoids
    re-sending a parameter the model already rejected.
    """
    effort = config.thinking_effort
    if effort is not ThinkingEffort.NONE and effort_supported is not False:
        settings: ModelSettings | None = build_model_settings(model, effort)
    else:
        settings = None

    ep_settings = endpoint_model_settings(model_name)
    if ep_settings is not None:
        settings = {**(settings or {}), **ep_settings}

    return settings


_CONTEXT_OVERFLOW_HINT = "Try /compact to free context space, then re-send your message."

# Keywords in API error messages that indicate a context-length issue.
_OVERFLOW_KEYWORDS = (
    "context length",
    "context window",
    "maximum context",
    "token limit",
    "too many tokens",
    "too long",
    "max_tokens",
    "prompt is too long",
    "input is too long",
    "request too large",
    "content_too_large",
)

# Keywords paired with "effort" that signal the parameter was rejected.
_EFFORT_REJECTION_KEYWORDS = (
    "not support",
    "unsupported",
    "not available",
    "not allowed",
    "unknown parameter",
    "invalid parameter",
    "unrecognized",
)


def _is_effort_unsupported(exc: ModelHTTPError) -> bool:
    """Does the API error indicate that the model doesn't support the effort parameter?"""
    msg = str(exc).lower()
    return "effort" in msg and any(kw in msg for kw in _EFFORT_REJECTION_KEYWORDS)


def _is_context_overflow(exc: ModelHTTPError) -> bool:
    """Heuristic: does the API error indicate a context-length problem?"""
    if exc.status_code == 413:
        return True
    if exc.status_code == 400:
        msg = str(exc).lower()
        return any(kw in msg for kw in _OVERFLOW_KEYWORDS)
    return False


_LIMIT_SUMMARY_PROMPT = (
    "You have reached the tool-call limit for this turn. "
    "Summarize what you accomplished so far and what remains to be done, "
    "so the user can decide whether to ask you to continue."
)


def handle_llm(engine: Engine, message: str) -> None:
    """Send a message to the active LLM, streaming the response."""
    if not engine.state.has_llm:
        engine._warn("No LLM connected. Use /connect claude, chatgpt, or openai.")
        return

    try:
        model = build_model(engine.state.model_name)
    except RbtrError as e:
        engine._warn(str(e))
        return

    try:
        _run_agent(engine, model, message)
    except ModelHTTPError as exc:
        if exc.status_code == 400 and _is_effort_unsupported(exc):
            engine.state.effort_supported = False
            engine._out("Model does not support effort — retrying without it…")
            _run_agent(engine, model, message)
            return
        if exc.status_code == 400 and is_history_format_error(exc):
            engine._out("Retrying with simplified history…")
            _run_agent(engine, model, message, simplify_history=True)
            return
        if _is_context_overflow(exc):
            _auto_compact_on_overflow(engine, message)
            return
        raise


def _auto_compact_on_overflow(engine: Engine, message: str) -> None:
    """Compact history and retry after a context-overflow error.

    ``compact_history`` handles all edge cases internally (no LLM,
    too few messages, LLM failure).  If compaction doesn't help,
    the retry raises the same overflow and the caller handles it.
    """
    from .compact import compact_history  # deferred: avoid circular at module level

    engine._out("Context limit reached — compacting history…")
    compact_history(engine)
    engine._out("Retrying with compacted history…")
    handle_llm(engine, message)


# ── Agent run ────────────────────────────────────────────────────────


def _run_agent(
    engine: Engine,
    model: Model,
    message: str,
    *,
    simplify_history: bool = False,
) -> None:
    """Stream an agent run, blocking until complete."""
    future = asyncio.run_coroutine_threadsafe(
        _stream_agent(engine, model, message, simplify_history=simplify_history),
        engine._loop,
    )
    future.result()


async def _stream_agent(
    engine: Engine,
    model: Model,
    message: str,
    *,
    simplify_history: bool = False,
) -> None:
    """Run the agent with cancellation support, update session state."""
    # Load history from DB — the source of truth.
    store = engine.store
    history = store.load_messages(engine.state.session_id)
    history, repaired_tools = _repair_dangling_tool_calls(history)
    if repaired_tools:
        names = ", ".join(repaired_tools)
        engine._warn(
            f"Previous turn was cancelled mid-tool-call ({names}). "
            f"Those tool results are lost — the model will continue without them."
        )
    if simplify_history:
        history = demote_thinking(history)
    deps = AgentDeps(state=engine.state)

    settings = resolve_model_settings(
        model, engine.state.model_name, effort_supported=engine.state.effort_supported
    )
    if (
        config.thinking_effort is not ThinkingEffort.NONE
        and engine.state.effort_supported is not False
    ):
        engine.state.effort_supported = settings is not None

    engine.state.usage.snapshot_base()

    # ── inner helper: one agent.iter() pass ──────────────────────

    async def _do_stream(
        prompt: str | None,
        msg_history: list[ModelMessage],
    ) -> _StreamResult:
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
                    if isinstance(node, ModelRequestNode):
                        writer = store.begin_response(
                            engine.state.session_id,
                            model_name=engine.state.model_name,
                        )
                        last_writer = writer

                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                match event:
                                    case PartStartEvent(index=idx, part=part):
                                        writer.add_part(idx, part)
                                    case PartDeltaEvent(delta=delta):
                                        if isinstance(delta, TextPartDelta):
                                            engine._emit(TextDelta(delta=delta.content_delta))
                                    case PartEndEvent(index=idx, part=part):
                                        writer.finish_part(idx, part)

                        all_msgs = run.all_messages()

                        # Extract per-request token counts from the
                        # ModelResponse that was just streamed.
                        last_resp = all_msgs[-1] if all_msgs else None
                        if isinstance(last_resp, ModelResponse):
                            writer.finish(
                                input_tokens=last_resp.usage.input_tokens,
                                output_tokens=last_resp.usage.output_tokens,
                                cache_read_tokens=last_resp.usage.cache_read_tokens or None,
                                cache_write_tokens=last_resp.usage.cache_write_tokens or None,
                            )
                        else:
                            writer.finish()

                        n = _save_new_requests(engine, all_msgs, saved_count)
                        saved_request_count += n
                        saved_count = len(all_msgs)

                    elif isinstance(node, CallToolsNode):
                        _update_live_usage(engine, run.usage(), node.model_response)
                        has_tool_calls = any(
                            isinstance(p, ToolCallPart) for p in node.model_response.parts
                        )
                        async with node.stream(run.ctx) as tool_stream:
                            async for tool_event in tool_stream:
                                _emit_tool_event(engine, tool_event)

                        all_msgs = run.all_messages()
                        n = _save_new_requests(engine, all_msgs, saved_count)
                        saved_request_count += n
                        saved_count = len(all_msgs)

                        # Mid-turn compaction: only when the model made
                        # tool calls (more requests will follow).  When
                        # there are no tool calls the turn ends naturally.
                        if (
                            has_tool_calls
                            and engine.state.usage.context_used_pct
                            >= config.compaction.auto_compact_pct
                        ):
                            compact_needed = True
                            break
            except UsageLimitExceeded:
                limit_hit = True

        new = list(run.new_messages())
        requests: list[ModelMessage] = [m for m in new if isinstance(m, ModelRequest)]
        if len(requests) > saved_request_count:
            _save_messages_safe(engine, requests[saved_request_count:])

        return _StreamResult(
            all_messages=list(run.all_messages()),
            new_messages=new,
            usage=run.usage(),
            limit_hit=limit_hit,
            last_writer=last_writer,
            compact_needed=compact_needed,
        )

    # ── run with cancellation ────────────────────────────────────

    result = await _run_with_cancel(engine, _do_stream(message, history))

    # Mid-turn compaction: if context exceeded the threshold during
    # a tool-call cycle, compact once and resume.
    if result.compact_needed:
        from .compact import compact_history_async  # deferred: avoid circular

        engine._out("Context limit reached mid-turn — compacting…")
        await compact_history_async(
            engine, extra_instructions="The model is mid-turn with active tool calls."
        )
        # Reload from DB after compaction.
        history = store.load_messages(engine.state.session_id)
        history, _ = _repair_dangling_tool_calls(history)
        engine.state.usage.snapshot_base()

        resume = await _run_with_cancel(engine, _do_stream(None, history))
        result = _merge_results(result, resume)

    # Record usage and set cost on the last response.
    run_cost, cost_available = _record_usage(engine, result.new_messages, result.usage)
    if result.last_writer is not None:
        # Find the last ModelResponse to get per-request token counts.
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

    if result.limit_hit:
        await _stream_summary(engine, model, settings)

    # Post-turn compaction for turns that didn't trigger mid-turn.
    if (
        not result.compact_needed
        and engine.state.usage.context_used_pct >= config.compaction.auto_compact_pct
    ):
        from .compact import compact_history_async  # deferred: avoid circular at module level

        await compact_history_async(engine)


async def _stream_summary(
    engine: Engine,
    model: Model,
    settings: ModelSettings | None,
) -> None:
    """Ask the model to summarize progress after hitting the tool-call limit.

    Uses a plain agent (no tools) with a single request so the model
    can only produce text — no further tool calls.
    """
    history = engine.store.load_messages(engine.state.session_id)
    history, _ = _repair_dangling_tool_calls(history)

    summary_agent: Agent[None, str] = Agent(model)
    async with summary_agent.iter(
        _LIMIT_SUMMARY_PROMPT,
        model=model,
        message_history=history or None,
        model_settings=settings,
        usage_limits=UsageLimits(request_limit=1),
    ) as run:
        async for node in run:
            if isinstance(node, ModelRequestNode):
                async with node.stream(run.ctx) as stream:
                    async for delta in stream.stream_text(delta=True):
                        engine._emit(TextDelta(delta=delta))

    new_summary = list(run.new_messages())

    # Persist summary messages.
    _save_messages_safe(engine, new_summary)


# ── Persistence helpers ──────────────────────────────────────────────


def _save_new_requests(
    engine: Engine,
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
        _save_messages_safe(engine, requests)
    return len(requests)


def _save_messages_safe(engine: Engine, messages: list[ModelMessage]) -> None:
    """Save messages to the store, logging failures instead of raising.

    Relies on ``engine._sync_store_context()`` having been called at
    task start — metadata is inherited from the stored context.
    """
    if not messages:
        return
    try:
        engine.store.save_messages(engine.state.session_id, messages)
    except OSError:
        log.warning("sessions: failed to persist messages", exc_info=True)


async def _run_with_cancel(
    engine: Engine,
    coro: collections.abc.Coroutine[object, object, _StreamResult],
) -> _StreamResult:
    """Run *coro* with a parallel cancellation watcher.

    Raises ``TaskCancelled`` if the engine's cancel event fires
    before the coroutine completes.
    """

    async def _watch() -> None:
        while not engine._cancel.is_set():
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
    engine: Engine,
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
            engine._emit(ToolCallStarted(tool_name=part.tool_name, args=args_str))
        case FunctionToolResultEvent(result=result):
            if isinstance(result, ToolReturnPart):
                text = str(result.content)
                max_chars = config.tui.tool_max_chars
                if len(text) > max_chars:
                    text = text[:max_chars] + "…"
            else:
                text = "(retry)"
            tool_name = getattr(result, "tool_name", None) or "?"
            engine._emit(ToolCallFinished(tool_name=tool_name, result=text))


# ── Usage tracking ───────────────────────────────────────────────────


def _update_live_usage(
    engine: Engine,
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
    usage = engine.state.usage
    base = usage.live_base
    usage.input_tokens = base.input_tokens + run_usage.input_tokens
    usage.output_tokens = base.output_tokens + run_usage.output_tokens
    usage.cache_read_tokens = base.cache_read_tokens + run_usage.cache_read_tokens
    usage.cache_write_tokens = base.cache_write_tokens + run_usage.cache_write_tokens
    usage.last_input_tokens = response.usage.input_tokens

    # Set context window from model metadata so the footer shows the
    # correct value during streaming — works for both endpoints and
    # built-in providers (Anthropic, OpenAI, etc.).
    _apply_model_context_window(engine)


def _apply_model_context_window(engine: Engine) -> None:
    """Set the context window from model metadata if available.

    No-op when the context window is already known.  Checks endpoint
    metadata first, then falls back to genai-prices for built-in
    providers.
    """
    if engine.state.usage.context_window_known:
        return
    ctx = model_context_window(engine.state.model_name)
    if ctx is not None:
        engine.state.usage.context_window = ctx
        engine.state.usage.context_window_known = True


def _record_usage(
    engine: Engine,
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
    meta_context_window = model_context_window(engine.state.model_name)
    if meta_context_window is not None:
        context_window = meta_context_window

    response_count = sum(1 for m in new_messages if isinstance(m, ModelResponse))
    engine.state.usage.record_run(
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
