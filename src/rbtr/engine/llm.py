"""Handler for LLM queries — streaming via the shared agent."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    HandleResponseEvent,
    ModelMessage,
    ModelResponse,
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
from .save import save_new_messages
from .types import TaskCancelled

if TYPE_CHECKING:
    from .core import Engine


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
    if not engine.session.has_llm:
        engine._warn("No LLM connected. Use /connect claude, chatgpt, or openai.")
        return

    try:
        model = build_model(engine.session.model_name)
    except RbtrError as e:
        engine._warn(str(e))
        return

    try:
        _run_agent(engine, model, message)
    except ModelHTTPError as exc:
        if exc.status_code == 400 and _is_effort_unsupported(exc):
            engine.session.effort_supported = False
            engine._out("Model does not support effort — retrying without it…")
            _run_agent(engine, model, message)
            return
        if exc.status_code == 400 and is_history_format_error(exc):
            engine._out("Retrying with simplified history…")
            engine.session.message_history = demote_thinking(engine.session.message_history)
            _run_agent(engine, model, message)
            return
        if _is_context_overflow(exc) and _auto_compact_on_overflow(engine, message):
            return
        if _is_context_overflow(exc):
            engine._out(_CONTEXT_OVERFLOW_HINT)
        raise


def _auto_compact_on_overflow(engine: Engine, message: str) -> bool:
    """Attempt auto-compaction after a context-overflow error.

    Returns True if compaction succeeded and the message was re-sent,
    False if compaction was not possible (caller should fall through
    to the normal error path).
    """
    if len(engine.session.message_history) < 2:  # nothing to compact
        return False

    from .compact import compact_history  # deferred: avoid circular at module level

    pre_len = len(engine.session.message_history)
    engine._out("Context limit reached — compacting history…")
    compact_history(engine)

    if len(engine.session.message_history) >= pre_len:
        # Compaction didn't reduce anything — don't retry.
        return False

    engine._out("Retrying with compacted history…")
    handle_llm(engine, message)
    return True


def _run_agent(engine: Engine, model: Model, message: str) -> None:
    """Stream an agent run, blocking until complete."""
    future = asyncio.run_coroutine_threadsafe(_stream_agent(engine, model, message), engine._loop)
    future.result()


async def _stream_agent(engine: Engine, model: Model, message: str) -> None:
    """Run the agent with cancellation support, update session state."""
    history = engine.session.message_history
    deps = AgentDeps(session=engine.session)

    settings = resolve_model_settings(
        model, engine.session.model_name, effort_supported=engine.session.effort_supported
    )
    if (
        config.thinking_effort is not ThinkingEffort.NONE
        and engine.session.effort_supported is not False
    ):
        engine.session.effort_supported = settings is not None

    engine.session.usage.snapshot_base()

    async def _do_stream() -> tuple[list[ModelMessage], RunUsage, bool]:
        limit_hit = False
        async with agent.iter(
            message,
            model=model,
            deps=deps,
            message_history=history or None,
            model_settings=settings,
            usage_limits=UsageLimits(request_limit=config.tools.max_requests_per_turn),
        ) as run:
            try:
                async for node in run:
                    if isinstance(node, ModelRequestNode):
                        async with node.stream(run.ctx) as stream:
                            async for delta in stream.stream_text(delta=True):
                                engine._emit(TextDelta(delta=delta))
                    elif isinstance(node, CallToolsNode):
                        # model_response carries per-request usage from
                        # the request that just completed — update the
                        # session so the footer shows live progress.
                        _update_live_usage(engine, run.usage(), node.model_response)
                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                _emit_tool_event(engine, event)
            except UsageLimitExceeded:
                limit_hit = True
        return list(run.all_messages()), run.usage(), limit_hit

    async def _watch_cancel() -> None:
        while not engine._cancel.is_set():
            await asyncio.sleep(0.1)
        raise TaskCancelled

    stream_task = asyncio.create_task(_do_stream())
    cancel_task = asyncio.create_task(_watch_cancel())

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

    messages, run_usage, limit_hit = stream_task.result()
    engine.session.message_history = list(messages)
    _record_usage(engine, history, messages, run_usage)

    if limit_hit:
        await _stream_summary(engine, model, settings)

    # Auto-compact when context usage exceeds the threshold.
    if engine.session.usage.context_used_pct >= config.compaction.auto_compact_pct:
        from .compact import compact_history  # deferred: avoid circular at module level

        compact_history(engine)


async def _stream_summary(
    engine: Engine,
    model: Model,
    settings: ModelSettings | None,
) -> None:
    """Ask the model to summarize progress after hitting the tool-call limit.

    Uses a plain agent (no tools) with a single request so the model
    can only produce text — no further tool calls.
    """
    history = engine.session.message_history

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

    summary_messages = list(run.all_messages())
    engine.session.message_history = list(history) + summary_messages[len(history) :]


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

    Does **not** increment ``message_count`` or cost — the
    authoritative ``_record_usage`` at run-end handles those.
    """
    usage = engine.session.usage
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
    if engine.session.usage.context_window_known:
        return
    ctx = model_context_window(engine.session.model_name)
    if ctx is not None:
        engine.session.usage.context_window = ctx
        engine.session.usage.context_window_known = True


def _record_usage(
    engine: Engine,
    old_history: list[ModelMessage],
    messages: list[ModelMessage],
    run_usage: RunUsage,
) -> None:
    """Extract cost/context-window from new messages and update session usage.

    ``run_usage.input_tokens`` is the **sum** across all requests in
    the run (PydanticAI accumulates).  For context-% we need the
    *last* request's input tokens — that's the actual prompt size the
    model received, reflecting the full conversation + tool results.
    We pull it from the last ``ModelResponse.usage.input_tokens``.
    """
    new_messages = messages[len(old_history or []) :]
    run_cost = 0.0
    cost_available = False
    context_window: int | None = None
    last_input_tokens: int | None = None

    for msg in new_messages:
        if not isinstance(msg, ModelResponse) or not msg.model_name:
            continue
        # Each ModelResponse carries per-request usage — the last one
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
    meta_context_window = model_context_window(engine.session.model_name)
    if meta_context_window is not None:
        context_window = meta_context_window

    engine.session.usage.record_run(
        input_tokens=run_usage.input_tokens,
        output_tokens=run_usage.output_tokens,
        cache_read_tokens=run_usage.cache_read_tokens,
        cache_write_tokens=run_usage.cache_write_tokens,
        last_input_tokens=last_input_tokens,
        cost=run_cost,
        cost_available=cost_available,
        context_window=context_window,
    )

    # Persist new messages to the session store.
    save_new_messages(engine, run_cost=run_cost if cost_available else None)
