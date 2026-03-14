"""Cross-session memory — fact extraction and deduplication.

Extracts durable facts from conversation messages using a tool-less
PydanticAI agent with structured output.  Each extracted fact is
tagged by the LLM as ``new``, ``confirm``, or ``supersede`` — the
LLM sees all existing facts in the prompt and makes the dedup
decision itself.

The agent follows the same pattern as ``compact_agent`` and the
main ``agent``: module-level ``Agent()``, ``@instructions``
decorators for the static task prompt, model passed at each call
site.

Public API
----------
- ``extract_facts_from_ctx`` — background-thread entry point.
- ``apply_fact_extraction`` — async orchestrator (process, persist,
  clarify).  Used by both ``extract_facts_from_ctx`` and
  ``compact.py``.
- ``process_extracted_facts`` — apply extracted facts to the store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from rbtr.config import config
from rbtr.events import FactExtractionFinished, FactExtractionStarted
from rbtr.exceptions import RbtrError
from rbtr.llm.context import LLMContext
from rbtr.llm.history import serialise_for_summary
from rbtr.llm.usage import extract_cost
from rbtr.prompts import render_existing_facts, render_fact_extraction, render_system
from rbtr.providers import build_model, model_settings
from rbtr.sessions.kinds import GLOBAL_SCOPE, FragmentKind
from rbtr.sessions.overhead import FactExtractionOverhead, FactExtractionSource
from rbtr.sessions.store import Fact, SessionStore

log = logging.getLogger(__name__)

# ── Structured output ────────────────────────────────────────────────


class FactAction(StrEnum):
    """What to do with an extracted fact."""

    NEW = "new"
    CONFIRM = "confirm"
    SUPERSEDE = "supersede"


class ExtractedFact(BaseModel):
    """A single fact extracted by the LLM."""

    content: str
    scope: str = "repo"
    """``'global'`` or ``'repo'``."""
    action: FactAction = FactAction.NEW
    existing_content: str | None = None
    """Content of the existing fact when ``action`` is ``confirm`` or ``supersede``."""


class FactExtractionResult(BaseModel):
    """Structured output from the fact extraction agent."""

    facts: list[ExtractedFact] = []


# ── Extraction agent ─────────────────────────────────────────────────


@dataclass
class FactExtractionDeps:
    """Dependencies injected into every fact extraction run."""

    existing_facts: list[Fact]


fact_extract_agent: Agent[FactExtractionDeps, FactExtractionResult] = Agent(
    output_type=FactExtractionResult,
    deps_type=FactExtractionDeps,
)


@fact_extract_agent.instructions
def _system() -> str:
    """Shared system prompt — same identity as the main agent."""
    return render_system()


@fact_extract_agent.instructions
def _fact_extraction_task() -> str:
    """Task instructions — what to extract, format, rules."""
    return render_fact_extraction()


@fact_extract_agent.instructions
def _existing_facts(ctx: RunContext[FactExtractionDeps]) -> str:
    """Render existing facts as reference context for dedup decisions."""
    if not ctx.deps.existing_facts:
        return ""
    return render_existing_facts([f.content for f in ctx.deps.existing_facts])


def _build_clarify_prompt(failed: list[ExtractedFact]) -> str:
    """Build a follow-up prompt asking the model to correct misquoted content.

    Sent as a continuation of the fact extraction conversation — the model
    already has the existing facts and conversation in its history.
    """
    parts: list[str] = [
        """\
The following facts you extracted had `existing_content` that
didn't exactly match any active fact. Correct the
`existing_content` to match an existing fact exactly, or
drop the fact if none match.""",
    ]
    for f in failed:
        parts.append(
            f"- action={f.action}, content={f.content!r}, existing_content={f.existing_content!r}"
        )
    return "\n".join(parts)


# ── User prompt builder ──────────────────────────────────────────────


def _build_user_prompt(conversation: str) -> str:
    """Build the user prompt from conversation text.

    Existing facts are injected via ``@fact_extract_agent.instructions``
    (through ``FactExtractionDeps``).  The user prompt contains only the
    conversation to analyse.
    """
    return f"## Conversation\n\n{conversation}"


# ── Fact injection ────────────────────────────────────────────────────


def render_facts_instruction(
    store: SessionStore,
    repo_scope: str | None,
    max_facts: int,
    max_tokens: int,
) -> str:
    """Render stored facts as a system instruction block.

    Loads active facts from global and repo scopes, most recently
    confirmed first.  Truncates to *max_facts* total and
    *max_tokens* estimated tokens.  Returns ``""`` if there are
    no facts or memory is disabled.
    """
    scopes = [GLOBAL_SCOPE]
    if repo_scope:
        scopes.append(repo_scope)

    facts: list[Fact] = []
    per_scope_limit = max_facts  # Each scope can contribute up to max_facts.
    for scope in scopes:
        facts.extend(store.load_active_facts(scope, limit=per_scope_limit))

    if not facts:
        return ""

    # Sort all facts together by last_confirmed_at DESC, take top N.
    facts.sort(key=lambda f: f.last_confirmed_at, reverse=True)
    facts = facts[:max_facts]

    # Build the instruction, truncating at the token cap.
    lines = ["## Facts\n"]
    token_count = _estimate_tokens(lines[0])

    for f in facts:
        scope_label = "global" if f.scope == GLOBAL_SCOPE else f.scope
        line = f"- [{scope_label}] {f.content}"
        line_tokens = _estimate_tokens(line)
        if token_count + line_tokens > max_tokens:
            break
        lines.append(line)
        token_count += line_tokens

    # Only the header — no facts fit within the token cap.
    if len(lines) == 1:
        return ""

    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token per 4 characters."""
    return len(text) // 4


# ── Fact processing ──────────────────────────────────────────────────


def _resolve_scope(extracted: ExtractedFact, repo_scope: str | None) -> str | None:
    """Map extracted fact scope to a store scope value.

    Returns ``None`` if repo scope is requested but no repo is connected.
    """
    if extracted.scope == GLOBAL_SCOPE:
        return GLOBAL_SCOPE
    return repo_scope


@dataclass
class ProcessResult:
    """Counts and fact IDs from ``process_extracted_facts``."""

    added: int = 0
    confirmed: int = 0
    superseded: int = 0
    fact_ids: list[str] = field(default_factory=list)
    failed: list[ExtractedFact] = field(default_factory=list)
    """Supersede/confirm facts whose ``existing_content`` didn't match."""


def process_extracted_facts(
    extracted: list[ExtractedFact],
    ctx: LLMContext,
) -> ProcessResult:
    """Apply extracted facts to the store.

    The LLM has already seen all existing facts in the fact extraction
    prompt and tagged each fact accordingly:

    - ``confirm``: bump the existing fact's ``last_confirmed_at``.
    - ``supersede``: insert new fact, mark old as superseded.
    - ``new``: insert as a new fact.

    No client-side dedup — the LLM makes the dedup decision.
    Long-term cleanup is handled by ``/memory purge``.

    Returns a ``ProcessResult`` with counts and touched fact IDs.
    """
    store = ctx.store
    session_id = ctx.state.session_id
    repo_scope = ctx.state.repo_scope
    result = ProcessResult()

    for ef in extracted:
        scope = _resolve_scope(ef, repo_scope)
        if scope is None:
            log.debug("memory: skipping repo-scoped fact (no repo): %s", ef.content[:60])
            continue

        if ef.action == FactAction.CONFIRM and ef.existing_content:
            existing = store.find_fact_by_content(ef.existing_content, scope)
            if existing:
                store.confirm_fact(existing.id)
                result.confirmed += 1
                result.fact_ids.append(existing.id)
            else:
                result.failed.append(ef)

        elif ef.action == FactAction.SUPERSEDE and ef.existing_content:
            existing = store.find_fact_by_content(ef.existing_content, scope)
            if existing:
                new_fact = store.insert_fact(scope, ef.content, session_id)
                store.supersede_fact(existing.id, new_fact.id)
                result.added += 1
                result.superseded += 1
                result.fact_ids.append(new_fact.id)
            else:
                result.failed.append(ef)

        else:
            # NEW or fallback — insert directly.
            new_fact = store.insert_fact(scope, ef.content, session_id)
            result.added += 1
            result.fact_ids.append(new_fact.id)

    return result


# ── Async fact extraction ────────────────────────────────────────────


@dataclass
class FactExtractionRun:
    """Raw result from a single fact extraction agent call."""

    facts: list[ExtractedFact]
    conversation_history: list[ModelMessage]
    input_tokens: int
    output_tokens: int
    cost: float
    model_name: str
    model: Model
    settings: ModelSettings | None
    deps: FactExtractionDeps


async def run_fact_extraction(
    messages: list[ModelMessage],
    store: SessionStore,
    repo_scope: str | None,
    model_name: str,
) -> FactExtractionRun | None:
    """Run the fact extraction agent and return raw results.

    Returns ``None`` if fact extraction is skipped (no messages, disabled,
    empty conversation, model error).  Does **not** process facts or
    persist overhead — callers handle that.
    """
    if not messages or not config.memory.enabled:
        return None

    conversation = serialise_for_summary(messages, max_tool_chars=500)
    if not conversation.strip():
        return None

    # Load existing facts — injected into instructions via deps.
    scopes = [GLOBAL_SCOPE]
    if repo_scope:
        scopes.append(repo_scope)
    per_scope_limit = config.memory.max_extraction_facts
    existing: list[Fact] = []
    for scope in scopes:
        existing.extend(store.load_active_facts(scope, limit=per_scope_limit))

    deps = FactExtractionDeps(existing_facts=existing)
    user_prompt = _build_user_prompt(conversation)

    extraction_model_name = config.memory.fact_extraction_model or model_name
    try:
        model = build_model(extraction_model_name)
    except RbtrError:
        log.warning("memory: cannot build model %r for extraction", extraction_model_name)
        return None

    effort = config.thinking_effort
    settings = model_settings(extraction_model_name, model, effort)

    try:
        result = await fact_extract_agent.run(
            user_prompt,
            deps=deps,
            model=model,
            model_settings=settings,
            usage_limits=UsageLimits(request_limit=1),
        )
    except (ModelHTTPError, OSError, RbtrError) as e:
        log.warning("memory: fact extraction failed: %s", e)
        return None

    usage = result.usage()
    cost = extract_cost(result.new_messages())

    return FactExtractionRun(
        facts=result.output.facts,
        conversation_history=result.all_messages(),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost=cost,
        model_name=extraction_model_name,
        model=model,
        settings=settings,
        deps=deps,
    )


# ── Clarification retry ──────────────────────────────────────────────


@dataclass
class _FactClarifyResult:
    """Corrected facts and overhead from a clarification call."""

    facts: list[ExtractedFact] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


_EMPTY_CLARIFY = _FactClarifyResult()


async def _clarify_failed_facts(
    failed: list[ExtractedFact],
    conversation_history: list[ModelMessage],
    model: Model,
    settings: ModelSettings | None,
    deps: FactExtractionDeps,
) -> _FactClarifyResult:
    """Ask the model to correct misquoted ``existing_content``.

    Continues the fact extraction conversation — the model already has
    the existing facts (via instructions) and conversation in its
    history.  A follow-up prompt lists the failures and asks for
    corrections.
    """
    prompt = _build_clarify_prompt(failed)

    try:
        result = await fact_extract_agent.run(
            prompt,
            deps=deps,
            model=model,
            model_settings=settings,
            message_history=conversation_history,
            usage_limits=UsageLimits(request_limit=1),
        )
    except (ModelHTTPError, OSError, RbtrError) as e:
        log.warning("memory: clarification failed: %s", e)
        return _EMPTY_CLARIFY

    c_usage = result.usage()
    c_cost = extract_cost(result.new_messages())
    corrected = result.output.facts

    return _FactClarifyResult(
        facts=corrected,
        input_tokens=c_usage.input_tokens,
        output_tokens=c_usage.output_tokens,
        cost=c_cost,
    )


# ── Orchestration ─────────────────────────────────────────────────────


async def apply_fact_extraction(
    ctx: LLMContext,
    run: FactExtractionRun,
    source: FactExtractionSource,
) -> ProcessResult:
    """Process a fact extraction run: apply facts, persist overhead, clarify.

    Single async orchestrator used by both ``extract_facts_from_ctx``
    (daemon thread, via ``portal.call``) and ``compact.py`` (already
    async).  Each step persists its overhead fragment immediately so
    work isn't lost on failure.

    Returns the accumulated ``ProcessResult``.
    """
    if not run.facts:
        _persist_overhead(
            ctx,
            FactExtractionOverhead(source=source, model_name=run.model_name or None),
            run.input_tokens,
            run.output_tokens,
            run.cost,
        )
        return ProcessResult()

    pr = process_extracted_facts(run.facts, ctx)
    _persist_overhead(
        ctx,
        FactExtractionOverhead(
            source=source,
            added=pr.added,
            confirmed=pr.confirmed,
            superseded=pr.superseded,
            model_name=run.model_name or None,
            fact_ids=pr.fact_ids,
        ),
        run.input_tokens,
        run.output_tokens,
        run.cost,
    )

    if pr.failed:
        ctx.out("Clarifying mismatched facts…")
        try:
            cr = await _clarify_failed_facts(
                pr.failed,
                run.conversation_history,
                run.model,
                run.settings,
                run.deps,
            )
        except Exception:
            log.exception("memory: clarification failed")
            cr = _EMPTY_CLARIFY

        if cr.facts:
            pr2 = process_extracted_facts(cr.facts, ctx)
            pr.added += pr2.added
            pr.confirmed += pr2.confirmed
            pr.superseded += pr2.superseded
            for f in pr2.failed:
                log.warning(
                    "memory: supersede/confirm still unresolved after retry: %s",
                    (f.existing_content or "")[:80],
                )

        if cr.input_tokens or cr.cost:
            _persist_overhead(
                ctx,
                FactExtractionOverhead(source=source, model_name=run.model_name or None),
                cr.input_tokens,
                cr.output_tokens,
                cr.cost,
            )

    return pr


# ── Background-thread entry point ────────────────────────────────────


def extract_facts_from_ctx(
    ctx: LLMContext,
    messages: list[ModelMessage],
    *,
    source: FactExtractionSource = FactExtractionSource.COMMAND,
) -> None:
    """Extract facts using an ``LLMContext``, emitting status events.

    Runs ``run_fact_extraction`` then delegates to ``apply_fact_extraction``
    for processing, clarification, and overhead persistence.

    Designed to be called from a background daemon thread via
    ``ctx.portal.call``.
    """
    if not config.memory.enabled:
        return

    model_name = ctx.state.model_name
    if not ctx.state.has_llm or not model_name:
        return

    ctx.emit(FactExtractionStarted())

    try:
        run = ctx.portal.call(
            run_fact_extraction,
            messages,
            ctx.store,
            ctx.state.repo_scope,
            model_name,
        )
    except Exception:
        log.exception("memory: fact extraction failed")
        ctx.emit(FactExtractionFinished())
        return

    if run is None:
        ctx.emit(FactExtractionFinished())
        return

    try:
        pr = ctx.portal.call(lambda: apply_fact_extraction(ctx, run, source))
    except Exception:
        log.exception("memory: fact extraction processing failed")
        ctx.emit(FactExtractionFinished())
        return

    ctx.emit(
        FactExtractionFinished(
            added=pr.added,
            confirmed=pr.confirmed,
            superseded=pr.superseded,
        )
    )


# ── Overhead persistence ─────────────────────────────────────────────


def _persist_overhead(
    ctx: LLMContext,
    payload: FactExtractionOverhead,
    input_tokens: int,
    output_tokens: int,
    cost: float,
) -> None:
    """Persist a fact extraction overhead fragment and record on usage."""
    if not input_tokens and not cost:
        return
    ctx.store.save_overhead(
        ctx.state.session_id,
        FragmentKind.OVERHEAD_FACT_EXTRACTION,
        payload,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
    )
    ctx.state.usage.record_fact_extraction(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
    )
