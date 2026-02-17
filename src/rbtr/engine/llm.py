"""Handler for LLM queries — streaming via the shared agent."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, cast

from pydantic_ai._agent_graph import ModelRequestNode
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from rbtr import RbtrError
from rbtr.events import TextDelta
from rbtr.providers import build_model

from .agent import AgentDeps, agent
from .history import demote_thinking, is_history_format_error
from .types import TaskCancelled

if TYPE_CHECKING:
    from .core import Engine


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
    history = cast(list[ModelMessage], engine.session.message_history)
    deps = AgentDeps(session=engine.session)

    async def _do_stream() -> tuple[list[ModelMessage], RunUsage]:
        async with agent.iter(
            message,
            model=model,
            deps=deps,
            message_history=history or None,
        ) as run:
            async for node in run:
                if isinstance(node, ModelRequestNode):
                    async with node.stream(run.ctx) as stream:
                        async for delta in stream.stream_text(delta=True):
                            engine._emit(TextDelta(delta=delta))
        return list(run.all_messages()), run.usage()

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

    messages, run_usage = stream_task.result()
    engine.session.message_history = list(messages)
    _record_usage(engine, history, messages, run_usage)


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
