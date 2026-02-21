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
from rbtr.providers import build_model, build_model_settings

from .agent import AgentDeps, agent
from .history import demote_thinking, is_history_format_error
from .types import TaskCancelled

if TYPE_CHECKING:
    from .core import Engine

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
        if exc.status_code != 400 or not is_history_format_error(exc):
            raise
        engine._out("Retrying with simplified history…")
        engine.session.message_history = demote_thinking(engine.session.message_history)
        _run_agent(engine, model, message)


def _run_agent(engine: Engine, model: Model, message: str) -> None:
    """Stream an agent run, blocking until complete."""
    future = asyncio.run_coroutine_threadsafe(_stream_agent(engine, model, message), engine._loop)
    future.result()


async def _stream_agent(engine: Engine, model: Model, message: str) -> None:
    """Run the agent with cancellation support, update session state."""
    history = engine.session.message_history
    deps = AgentDeps(session=engine.session)

    effort = config.thinking_effort
    if effort is not ThinkingEffort.NONE:
        settings = build_model_settings(model, effort)
        engine.session.effort_supported = settings is not None
    else:
        settings = None

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


def _record_usage(
    engine: Engine,
    old_history: list[ModelMessage],
    messages: list[ModelMessage],
    run_usage: RunUsage,
) -> None:
    """Extract cost/context-window from new messages and update session usage."""
    new_messages = messages[len(old_history or []) :]
    run_cost = 0.0
    cost_available = False
    context_window: int | None = None

    for msg in new_messages:
        if not isinstance(msg, ModelResponse) or not msg.model_name:
            continue
        try:
            price = msg.cost()
            run_cost += float(price.total_price)
            cost_available = True
            if price.model and price.model.context_window:
                context_window = price.model.context_window
        except (AssertionError, LookupError, ValueError):
            pass

    engine.session.usage.record_run(
        input_tokens=run_usage.input_tokens,
        output_tokens=run_usage.output_tokens,
        cache_read_tokens=run_usage.cache_read_tokens,
        cache_write_tokens=run_usage.cache_write_tokens,
        cost=run_cost,
        cost_available=cost_available,
        context_window=context_window,
    )
