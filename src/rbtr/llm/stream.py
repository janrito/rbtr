"""Handler for LLM queries — streaming via the shared agent."""

from __future__ import annotations

import collections.abc
import json
import logging
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import anyio
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
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
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits

from rbtr.config import ThinkingEffort, config
from rbtr.events import TextDelta, ToolCallFinished, ToolCallStarted
from rbtr.exceptions import RbtrError, TaskCancelled
from rbtr.providers import build_model, model_context_window, model_settings
from rbtr.sessions.scrub import scrub_secrets
from rbtr.sessions.serialise import (
    FailedAttempt,
    FailureKind,
    FragmentKind,
    FragmentStatus,
    HistoryRepair,
    IncidentOutcome,
    RecoveryStrategy,
)

from .agent import AgentDeps, agent
from .compact import compact_history, compact_history_async
from .context import LLMContext
from .errors import is_context_overflow, is_effort_unsupported
from .history import (
    consolidate_tool_returns,
    demote_thinking,
    flatten_tool_exchanges,
    format_tool_args,
    is_history_format_error,
    repair_dangling_tool_calls,
)

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


# ── Incident persistence ────────────────────────────────────────────────


def _persist_failed_request(ctx: LLMContext, message: str) -> str:
    """Persist the user's prompt as a failed ``REQUEST_MESSAGE``.

    Returns the message ID (used as ``turn_id`` in incident rows).
    """
    request = ModelRequest(parts=[UserPromptPart(content=message)])
    ids = ctx.store.save_messages(
        ctx.state.session_id,
        [request],
        status=FragmentStatus.FAILED,
    )
    return ids[0]


def _persist_failure(
    ctx: LLMContext,
    *,
    turn_id: str,
    failure_kind: FailureKind,
    strategy: RecoveryStrategy,
    exc: BaseException,
) -> str:
    """Persist an ``LLM_ATTEMPT_FAILED`` incident row.

    Returns the incident row ID.
    """
    payload = FailedAttempt(
        turn_id=turn_id,
        failure_kind=failure_kind,
        strategy=strategy,
        diagnostic=scrub_secrets("".join(traceback.format_exception(exc))),
        error_text=scrub_secrets(str(exc)[:500]),
        model_name=ctx.state.model_name,
        status_code=exc.status_code if isinstance(exc, ModelHTTPError) else None,
    )
    return ctx.store.save_incident(
        ctx.state.session_id,
        FragmentKind.LLM_ATTEMPT_FAILED,
        payload,
    )


def _persist_history_repair(
    ctx: LLMContext,
    payload: HistoryRepair,
) -> str:
    """Persist an ``LLM_HISTORY_REPAIR`` incident row.

    Returns the incident row ID.
    """
    return ctx.store.save_incident(
        ctx.state.session_id,
        FragmentKind.LLM_HISTORY_REPAIR,
        payload,
    )


def _retry_with_incidents(
    ctx: LLMContext,
    message: str,
    exc: BaseException,
    *,
    failure_kind: FailureKind,
    strategy: RecoveryStrategy,
    retry: collections.abc.Callable[[], None],
) -> None:
    """Persist incident records, run *retry*, and record the outcome.

    1. Persist the user's prompt as a failed ``REQUEST_MESSAGE``.
    2. Persist an ``LLM_ATTEMPT_FAILED`` incident row.
    3. Call *retry*.  On success → ``outcome = 'recovered'``.
       On exception → ``outcome = 'failed'``, then re-raise.
    """
    turn_id = _persist_failed_request(ctx, message)
    incident_id = _persist_failure(
        ctx,
        turn_id=turn_id,
        failure_kind=failure_kind,
        strategy=strategy,
        exc=exc,
    )
    try:
        retry()
    except Exception:
        ctx.store.update_incident_json(incident_id, "outcome", IncidentOutcome.FAILED)
        raise
    ctx.store.update_incident_json(incident_id, "outcome", IncidentOutcome.RECOVERED)


def handle_llm(ctx: LLMContext, message: str) -> None:
    """Send a message to the active LLM, streaming the response."""
    if not ctx.state.has_llm or not ctx.state.model_name:
        ctx.warn("No LLM connected. Use /connect claude, chatgpt, or openai.")
        return

    try:
        model = build_model(ctx.state.model_name)
    except RbtrError as e:
        ctx.warn(str(e))
        return

    history_repair_level = 0
    try:
        _run_agent(ctx, model, message)
    except ModelHTTPError as exc:
        if exc.status_code == HTTPStatus.BAD_REQUEST and is_effort_unsupported(exc):
            ctx.state.effort_supported = False
            ctx.out("Model does not support effort — retrying without it…")
            _retry_with_incidents(
                ctx,
                message,
                exc,
                failure_kind=FailureKind.EFFORT_UNSUPPORTED,
                strategy=RecoveryStrategy.EFFORT_OFF,
                retry=lambda: _run_agent(ctx, model, message),
            )
            return
        if exc.status_code == HTTPStatus.BAD_REQUEST and is_history_format_error(exc):
            next_level = history_repair_level + 1
            if next_level == 1:
                ctx.out("Retrying with consolidated tool returns…")
                strategy = RecoveryStrategy.CONSOLIDATE_TOOL_RETURNS
            else:
                ctx.out("Retrying with simplified history…")
                strategy = RecoveryStrategy.SIMPLIFY_HISTORY
            _retry_with_incidents(
                ctx,
                message,
                exc,
                failure_kind=FailureKind.HISTORY_FORMAT,
                strategy=strategy,
                retry=lambda: _run_agent(
                    ctx,
                    model,
                    message,
                    history_repair_level=next_level,
                ),
            )
            return
        if is_context_overflow(exc):
            _retry_with_incidents(
                ctx,
                message,
                exc,
                failure_kind=FailureKind.OVERFLOW,
                strategy=RecoveryStrategy.COMPACT_THEN_RETRY,
                retry=lambda: _auto_compact_on_overflow(ctx, message),
            )
            return
        if exc.status_code == HTTPStatus.NOT_FOUND:
            ctx.error_with_detail(
                f"Model not found: {ctx.state.model_name}. Check the model name with /model.",
                scrub_secrets(_format_http_error(exc)),
            )
            return
        ctx.error_with_detail(
            f"LLM request failed ({exc.status_code}): {_short_message(exc)}",
            scrub_secrets(_format_http_error(exc)),
        )
        return
    except ValueError as exc:
        if _is_tool_args_error(exc):
            ctx.out("Retrying with simplified history…")
            _retry_with_incidents(
                ctx,
                message,
                exc,
                failure_kind=FailureKind.TOOL_ARGS,
                strategy=RecoveryStrategy.SIMPLIFY_HISTORY,
                retry=lambda: _run_agent(ctx, model, message, history_repair_level=2),
            )
            return
        raise
    except UnexpectedModelBehavior as exc:
        log.info("UnexpectedModelBehavior during agent run", exc_info=True)
        ctx.error_with_detail(
            "Model returned an unusable response (e.g. only thinking, no text or tool calls). "
            "Try a different model or retry.",
            scrub_secrets("".join(traceback.format_exception(exc))),
        )
    except TypeError as exc:
        log.info("TypeError during agent run — retrying with simplified history", exc_info=True)
        ctx.out("Retrying with simplified history…")
        _retry_with_incidents(
            ctx,
            message,
            exc,
            failure_kind=FailureKind.TYPE_ERROR,
            strategy=RecoveryStrategy.SIMPLIFY_HISTORY,
            retry=lambda: _run_agent(ctx, model, message, history_repair_level=2),
        )
    except TimeoutError:
        timeout = config.tools.turn_timeout
        ctx.error(
            f"Turn timed out after {timeout:.0f}s. "
            f"Partial results have been saved. "
            f"Increase `tools.turn_timeout` in config or set to 0 to disable."
        )


def _short_message(exc: ModelHTTPError) -> str:
    """Extract a short human-readable message from an HTTP error."""
    body = str(exc.body) if exc.body else str(exc)
    # Trim to a reasonable length for the summary line.
    return body[:200] + "…" if len(body) > 200 else body


def _format_http_error(exc: ModelHTTPError) -> str:
    """Format a ``ModelHTTPError`` with full detail for expansion."""
    parts = [
        f"Status: {exc.status_code}",
        f"Model:  {exc.model_name}",
    ]
    if exc.body:
        try:
            body_str = json.dumps(exc.body, indent=2)
        except (TypeError, ValueError):
            body_str = str(exc.body)
        parts.append(f"Body:\n{body_str}")
    parts.append(f"\n{''.join(traceback.format_exception(exc))}")
    return "\n".join(parts)


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
                        writer = ctx.store.begin_response(
                            ctx.state.session_id,
                            model_name=ctx.state.model_name,
                        )
                        last_writer = writer

                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                match event:
                                    case PartStartEvent(index=idx, part=part):
                                        writer.add_part(idx, part)
                                        if isinstance(part, TextPart) and part.content:
                                            ctx.emit(TextDelta(delta=part.content))
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
    history_repair_level: int = 0,
) -> None:
    """Stream an agent run, blocking until complete."""
    ctx.portal.call(
        lambda: _run_with_cancel(
            ctx,
            _stream_agent(ctx, model, message, history_repair_level=history_repair_level),
        )
    )


def _prepare_turn(
    ctx: LLMContext,
    model: Model,
    history_repair_level: int,
) -> tuple[list[ModelMessage], AgentDeps, ModelSettings | None]:
    """Load history, repair dangling tool calls, resolve settings.

    *history_repair_level* controls how aggressively the history is
    adapted: 0 = repair only, 1 = consolidate tool returns,
    2 = demote thinking + flatten tool exchanges.

    Repairs are transient — applied in memory only.  The DB retains
    the original conversation.  Each manipulation is recorded as an
    ``LLM_HISTORY_REPAIR`` incident row.

    Mutates engine state: emits warnings for repaired tool calls,
    tracks ``effort_supported``, and snapshots the usage baseline
    for the upcoming run.
    """
    history = ctx.store.load_messages(ctx.state.session_id)
    history, repaired_tools = repair_dangling_tool_calls(history)
    if repaired_tools and history_repair_level == 0:
        names = ", ".join(repaired_tools)
        ctx.warn(
            f"Previous turn was cancelled mid-tool-call ({names}). "
            f"Those tool results are lost — the model will continue without them."
        )
        _persist_history_repair(
            ctx,
            HistoryRepair(
                strategy=RecoveryStrategy.REPAIR_DANGLING,
                tool_names=repaired_tools,
                call_count=len(repaired_tools),
                reason="cancelled_mid_tool_call",
            ),
        )
    if history_repair_level >= 1:
        consolidated = consolidate_tool_returns(history)
        history = consolidated.history
        if consolidated.turns_fixed:
            _persist_history_repair(
                ctx,
                HistoryRepair(
                    strategy=RecoveryStrategy.CONSOLIDATE_TOOL_RETURNS,
                    turns_fixed=consolidated.turns_fixed,
                    reason="cross_provider_retry",
                ),
            )

    if history_repair_level >= 2:
        demote = demote_thinking(history)
        history = demote.history
        if demote.parts_demoted:
            _persist_history_repair(
                ctx,
                HistoryRepair(
                    strategy=RecoveryStrategy.DEMOTE_THINKING,
                    parts_demoted=demote.parts_demoted,
                    reason="cross_provider_retry",
                ),
            )

        flatten = flatten_tool_exchanges(history)
        history = flatten.history
        if flatten.tool_calls_flattened or flatten.tool_returns_flattened:
            _persist_history_repair(
                ctx,
                HistoryRepair(
                    strategy=RecoveryStrategy.FLATTEN_TOOL_EXCHANGES,
                    tool_calls_flattened=flatten.tool_calls_flattened,
                    tool_returns_flattened=flatten.tool_returns_flattened,
                    retry_prompts_dropped=flatten.retry_prompts_dropped,
                    reason="cross_provider_retry",
                ),
            )

    deps = AgentDeps(state=ctx.state)
    effort = ThinkingEffort.NONE if ctx.state.effort_supported is False else config.thinking_effort
    settings = model_settings(ctx.state.model_name, model, effort)
    if effort is not ThinkingEffort.NONE:
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
    history_repair_level: int = 0,
) -> None:
    """Run the agent with cancellation support, update session state."""
    timeout = config.tools.turn_timeout
    if timeout > 0:
        with anyio.fail_after(timeout):
            await _stream_agent_inner(
                ctx, model, message, history_repair_level=history_repair_level
            )
    else:
        await _stream_agent_inner(ctx, model, message, history_repair_level=history_repair_level)


async def _stream_agent_inner(
    ctx: LLMContext,
    model: Model,
    message: str,
    *,
    history_repair_level: int = 0,
) -> None:
    """Inner turn logic — streaming, compaction, and limit handling."""
    history, deps, settings = _prepare_turn(ctx, model, history_repair_level)

    result = await _do_stream(ctx, model, deps, settings, message, history)

    # Mid-turn compaction: if context exceeded the threshold during
    # a tool-call cycle, compact once and resume.
    if result.compact_needed:
        await compact_history_async(
            ctx,
            extra_instructions="The model is mid-turn with active tool calls.",
        )
        # Reload from DB after compaction.
        history = ctx.store.load_messages(ctx.state.session_id)
        ctx.state.usage.snapshot_base()

        resume = await _do_stream(ctx, model, deps, settings, None, history)
        result = _merge_results(result, resume)

    _finalize_turn(ctx, result)

    if result.limit_hit:
        history = ctx.store.load_messages(ctx.state.session_id)
        summary = await _do_stream(ctx, model, deps, settings, _LIMIT_SUMMARY_PROMPT, history)
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


def _save_messages_safe(ctx: LLMContext, messages: list[ModelMessage]) -> list[str]:
    """Save messages to the store, logging failures instead of raising.

    Relies on ``engine._sync_store_context()`` having been called at
    task start — metadata is inherited from the stored context.

    Returns the list of message-level row IDs, or ``[]`` on failure.
    """
    if not messages:
        return []
    try:
        return ctx.store.save_messages(ctx.state.session_id, messages)
    except OSError:
        log.warning("sessions: failed to persist messages", exc_info=True)
        return []


async def _run_with_cancel(
    ctx: LLMContext,
    coro: collections.abc.Awaitable[None],
) -> None:
    """Run *coro* with a cancellation watcher using anyio cancel scopes.

    A parallel task awaits an ``anyio.Event`` that the UI thread sets
    via ``anyio.from_thread.run_sync(event.set, token=...)`` (zero-
    latency bridge).  When it fires, the shared cancel scope is
    cancelled, tearing down the streaming coroutine.  Raises
    ``TaskCancelled`` so the engine can report the cancellation.

    Exceptions from *coro* are captured and re-raised **after** the task
    group exits cleanly, so they propagate as bare exceptions rather
    than being wrapped in an ``ExceptionGroup``.
    """
    failure: BaseException | None = None
    cancelled = False

    cancel_event = anyio.Event()
    ctx.cancel_slot[0] = cancel_event
    # If the threading.Event was already set before we entered (race),
    # fire the anyio event immediately.
    if ctx.cancel.is_set():
        cancel_event.set()

    try:
        async with anyio.create_task_group() as tg:

            async def _watch() -> None:
                nonlocal cancelled
                await cancel_event.wait()
                cancelled = True
                tg.cancel_scope.cancel()

            tg.start_soon(_watch)
            try:
                await coro
            except anyio.get_cancelled_exc_class():
                if cancelled:
                    raise TaskCancelled from None
                raise
            except BaseException as exc:
                failure = exc
            # Cancel the watcher — either normal exit or captured failure.
            tg.cancel_scope.cancel()
    finally:
        ctx.cancel_slot[0] = None

    if failure is not None:
        raise failure


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
            ctx.emit(ToolCallStarted(tool_name=part.tool_name, args=format_tool_args(part.args)))
        case FunctionToolResultEvent(result=result):
            tool_name = getattr(result, "tool_name", None) or "?"
            if isinstance(result, ToolReturnPart):
                text = str(result.content)
                max_chars = config.tui.tool_max_chars
                if len(text) > max_chars:
                    text = text[:max_chars] + "…"
                ctx.emit(ToolCallFinished(tool_name=tool_name, result=text))
            else:
                ctx.emit(
                    ToolCallFinished(
                        tool_name=tool_name,
                        result="",
                        error=str(result.content),
                    )
                )


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
