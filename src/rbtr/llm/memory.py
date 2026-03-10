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
- ``extract_facts_async`` — extract facts from messages (async).
- ``extract_facts_from_ctx`` — background-thread entry point.
- ``process_extracted_facts`` — apply extractions to the store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.usage import UsageLimits

from rbtr.config import config
from rbtr.events import MemoryExtractionFinished, MemoryExtractionStarted
from rbtr.exceptions import RbtrError
from rbtr.llm.context import LLMContext
from rbtr.llm.history import serialise_for_summary
from rbtr.prompts import render_memory_extract, render_system
from rbtr.providers import build_model, model_settings
from rbtr.sessions.store import Fact, SessionStore

log = logging.getLogger(__name__)

# ── Structured output ────────────────────────────────────────────────

GLOBAL_SCOPE = "global"


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
    existing_id: str | None = None
    """Set when ``action`` is ``confirm`` or ``supersede``."""


class ExtractionResult(BaseModel):
    """Structured output from the extraction agent."""

    facts: list[ExtractedFact] = []


@dataclass
class ExtractionOutcome:
    """Result of an extraction run, including overhead cost."""

    added: int = 0
    confirmed: int = 0
    superseded: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


# ── Extraction agent ─────────────────────────────────────────────────

extract_agent: Agent[None, ExtractionResult] = Agent(
    output_type=ExtractionResult,
)


@extract_agent.instructions
def _system() -> str:
    """Shared system prompt — same identity as the main agent."""
    return render_system()


@extract_agent.instructions
def _extraction_task() -> str:
    """Task instructions — what to extract, format, rules."""
    return render_memory_extract()


# ── User prompt builder ──────────────────────────────────────────────


def _build_user_prompt(conversation: str, existing_facts: list[Fact]) -> str:
    """Build the user prompt from conversation and existing facts.

    The static task instructions live in ``@extract_agent.instructions``.
    This function builds the per-call user prompt with the dynamic data.
    """
    parts: list[str] = ["## Existing facts\n"]
    if existing_facts:
        for f in existing_facts:
            parts.append(f"- [id={f.id}] {f.content}")
    else:
        parts.append("(none)")
    parts.append("\n## Conversation\n")
    parts.append(conversation)
    return "\n".join(parts)


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
    lines = ["## Learned facts\n"]
    token_count = _estimate_tokens(lines[0])

    for f in facts:
        scope_label = "global" if f.scope == GLOBAL_SCOPE else f.scope
        line = f"- [id={f.id}, {scope_label}] {f.content}"
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
    """Map extraction scope to a store scope value.

    Returns ``None`` if repo scope is requested but no repo is connected.
    """
    if extracted.scope == GLOBAL_SCOPE:
        return GLOBAL_SCOPE
    return repo_scope


def process_extracted_facts(
    extracted: list[ExtractedFact],
    store: SessionStore,
    session_id: str,
    repo_scope: str | None,
) -> tuple[int, int, int]:
    """Apply extracted facts to the store.

    The LLM has already seen all existing facts in the extraction
    prompt and tagged each extraction accordingly:

    - ``confirm``: bump the existing fact's ``last_confirmed_at``.
    - ``supersede``: insert new fact, mark old as superseded.
    - ``new``: insert as a new fact.

    No client-side dedup — the LLM makes the dedup decision.
    If it occasionally misses a near-duplicate, the hard-limit
    pruning keeps the store bounded.

    Returns ``(added, confirmed, superseded)`` counts.
    """
    added = 0
    confirmed = 0
    superseded = 0

    for ef in extracted:
        scope = _resolve_scope(ef, repo_scope)
        if scope is None:
            log.debug("memory: skipping repo-scoped fact (no repo): %s", ef.content[:60])
            continue

        if ef.action == FactAction.CONFIRM and ef.existing_id:
            store.confirm_fact(ef.existing_id)
            confirmed += 1

        elif ef.action == FactAction.SUPERSEDE and ef.existing_id:
            new_fact = store.insert_fact(scope, ef.content, session_id)
            store.supersede_fact(ef.existing_id, new_fact.id)
            added += 1
            superseded += 1

        else:
            # NEW or fallback — insert directly.
            store.insert_fact(scope, ef.content, session_id)
            added += 1

    return added, confirmed, superseded


# ── Async extraction ─────────────────────────────────────────────────


_EMPTY_OUTCOME = ExtractionOutcome()


async def extract_facts_async(
    messages: list[ModelMessage],
    store: SessionStore,
    session_id: str,
    model_name: str,
    repo_scope: str | None,
) -> ExtractionOutcome:
    """Extract facts from *messages* and apply to the store.

    Uses the extraction agent to identify durable facts.  The LLM
    sees all existing facts in the user prompt and tags each
    extraction as ``new``, ``confirm``, or ``supersede``.

    Returns an ``ExtractionOutcome`` with counts and overhead cost.
    """
    if not messages:
        return _EMPTY_OUTCOME

    if not config.memory.enabled:
        return _EMPTY_OUTCOME

    conversation = serialise_for_summary(messages, max_tool_chars=500)
    if not conversation.strip():
        return _EMPTY_OUTCOME

    # Load existing facts for the user prompt context.
    scopes = [GLOBAL_SCOPE]
    if repo_scope:
        scopes.append(repo_scope)
    existing: list[Fact] = []
    for scope in scopes:
        existing.extend(store.load_active_facts(scope))

    user_prompt = _build_user_prompt(conversation, existing)

    extraction_model_name = config.memory.extraction_model or model_name
    try:
        model = build_model(extraction_model_name)
    except RbtrError:
        log.warning("memory: cannot build model %r for extraction", extraction_model_name)
        return _EMPTY_OUTCOME

    effort = config.thinking_effort
    settings = model_settings(extraction_model_name, model, effort)

    try:
        result = await extract_agent.run(
            user_prompt,
            model=model,
            model_settings=settings,
            usage_limits=UsageLimits(request_limit=1),
        )
        extraction = result.output
    except (ModelHTTPError, OSError, RbtrError) as e:
        log.warning("memory: extraction failed: %s", e)
        return _EMPTY_OUTCOME

    # Extract overhead cost from the agent result.
    usage = result.usage()
    cost = 0.0
    for msg in result.new_messages():
        if isinstance(msg, ModelResponse) and msg.model_name:
            try:
                price = msg.cost()
                cost += float(price.total_price)
            except (AssertionError, LookupError, ValueError):
                pass

    if not extraction.facts:
        return ExtractionOutcome(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost=cost,
        )

    added, confirmed, superseded = process_extracted_facts(
        extraction.facts, store, session_id, repo_scope
    )
    return ExtractionOutcome(
        added=added,
        confirmed=confirmed,
        superseded=superseded,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost=cost,
    )


# ── Background-thread entry point ────────────────────────────────────


def extract_facts_from_ctx(
    ctx: LLMContext,
    messages: list[ModelMessage],
) -> None:
    """Extract facts using an ``LLMContext``, emitting status events.

    Designed to be called from a background daemon thread via
    ``ctx.portal.call``.
    """
    if not config.memory.enabled:
        return

    if not ctx.state.has_llm or not ctx.state.model_name:
        return

    repo_scope = ctx.state.repo_scope

    ctx.emit(MemoryExtractionStarted())

    try:
        outcome: ExtractionOutcome = ctx.portal.call(
            lambda: extract_facts_async(
                messages=messages,
                store=ctx.store,
                session_id=ctx.state.session_id,
                model_name=ctx.state.model_name,
                repo_scope=repo_scope,
            )
        )
    except Exception:
        log.exception("memory: extraction failed")
        ctx.emit(MemoryExtractionFinished())
        return

    ctx.emit(
        MemoryExtractionFinished(
            added=outcome.added,
            confirmed=outcome.confirmed,
            superseded=outcome.superseded,
        )
    )
