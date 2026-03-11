"""Serialisation for session persistence.

Translates between PydanticAI message types and ``Fragment``
objects for the ``fragments`` table.

Design
------
Each ``ModelMessage`` produces **1 + N** rows:

- **1 message row** (``message_id = id``, ``fragment_index = 0``):
  stores message-level metadata (everything except ``parts``).
- **N content rows** (one per part, ``fragment_index`` ≥ 1):
  stores each PydanticAI part as-is via its own ``TypeAdapter``.

No custom intermediate models.  Message headers are serialised via
PydanticAI's ``TypeAdapter[ModelMessage]`` (parts stripped).  Parts
are serialised directly via their built-in ``part_kind`` discriminator.

All functions are pure — no I/O, no engine or UI imports.

Tool-call args validation
~~~~~~~~~~~~~~~~~~~~~~~~~

``ToolCallPart.args`` is typed ``str | dict[str, Any] | None``.
PydanticAI accepts *any* string at validation time — it never
checks that a string is valid JSON.  Provider adapters only
discover malformed args later, when they call ``args_as_dict()``
(via ``pydantic_core.from_json``), which raises ``ValueError``.

A model can produce invalid args during streaming (e.g. mixed
XML/JSON fragments when it hallucinates the tool-call format).
The corrupt part is saved with ``status = 'complete'`` because
``_part_ta.dump_json`` faithfully serialises any string.

``reconstruct_message`` calls ``_validate_tool_call_args`` after
Pydantic validation.  Rather than raising (which would skip the
entire message and orphan its matching ``ToolReturnPart``\\s),
it **repairs** corrupt args to ``{}`` in-place.  The message
stays in the history so tool-call / tool-return pairing is
preserved and the provider adapter never rejects the history.
A secondary fallback in ``handle_llm`` catches any remaining
``ValueError`` and retries with simplified history
(``flatten_tool_exchanges`` converts tool calls to plain text,
bypassing ``args_as_dict()`` entirely).

Incident payloads
~~~~~~~~~~~~~~

LLM failure and history-repair records are persisted as
``Fragment`` rows with ``FragmentKind.LLM_ATTEMPT_FAILED`` or
``LLM_HISTORY_REPAIR``.  Their ``data_json`` column contains a
Pydantic ``BaseModel`` payload serialised via
``model_dump_json(exclude_none=True)``.  The models use
``extra="ignore"`` so payloads can evolve without migrations.
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Discriminator, TypeAdapter
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

log = logging.getLogger(__name__)

# ── Types ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Caller-supplied metadata written onto every row."""

    session_id: str
    session_label: str | None = None
    repo_owner: str | None = None
    repo_name: str | None = None
    model_name: str | None = None
    review_target: str | None = None


class FragmentKind(StrEnum):
    """Row kind for the ``fragments`` table.

    Five groups of values share a single TEXT column:

    - **Message-level**: ``REQUEST_MESSAGE``, ``RESPONSE_MESSAGE``
    - **PydanticAI parts**: values match ``part_kind`` strings so
      ``FragmentKind(part.part_kind)`` works for any content part
    - **User input**: ``COMMAND``, ``SHELL``
    - **Incidents**: ``LLM_ATTEMPT_FAILED``, ``LLM_HISTORY_REPAIR``
    - **Overhead**: ``OVERHEAD_COMPACTION``, ``OVERHEAD_FACT_EXTRACTION``

    Use ``is_message``, ``is_input``, ``is_incident``,
    ``is_overhead`` to test group membership.
    """

    # ── Message-level ────────────────────────────────────────────
    REQUEST_MESSAGE = "request-message"
    RESPONSE_MESSAGE = "response-message"

    # ── PydanticAI part kinds ────────────────────────────────────
    USER_PROMPT = "user-prompt"
    SYSTEM_PROMPT = "system-prompt"
    TOOL_RETURN = "tool-return"
    RETRY_PROMPT = "retry-prompt"
    TEXT = "text"
    TOOL_CALL = "tool-call"
    THINKING = "thinking"
    FILE = "file"
    BUILTIN_TOOL_CALL = "builtin-tool-call"
    BUILTIN_TOOL_RETURN = "builtin-tool-return"

    # ── User input ───────────────────────────────────────────────
    COMMAND = "command"
    SHELL = "shell"

    # ── Incidents ────────────────────────────────────────────────
    LLM_ATTEMPT_FAILED = "llm-attempt-failed"
    LLM_HISTORY_REPAIR = "llm-history-repair"

    # ── Overhead ─────────────────────────────────────────────────
    OVERHEAD_COMPACTION = "overhead-compaction"
    OVERHEAD_FACT_EXTRACTION = "overhead-fact-extraction"

    @property
    def is_message(self) -> bool:
        """True for message-level kinds (request / response)."""
        return self in _MESSAGE_KINDS

    @property
    def is_input(self) -> bool:
        """True for user input kinds (command / shell)."""
        return self in _INPUT_KINDS

    @property
    def is_incident(self) -> bool:
        """True for incident kinds (failures / repairs)."""
        return self in _INCIDENT_KINDS

    @property
    def is_overhead(self) -> bool:
        """True for overhead kinds (compaction / fact extraction cost)."""
        return self in _OVERHEAD_KINDS


_MESSAGE_KINDS = frozenset({FragmentKind.REQUEST_MESSAGE, FragmentKind.RESPONSE_MESSAGE})
_INPUT_KINDS = frozenset({FragmentKind.COMMAND, FragmentKind.SHELL})
_INCIDENT_KINDS = frozenset({FragmentKind.LLM_ATTEMPT_FAILED, FragmentKind.LLM_HISTORY_REPAIR})
_OVERHEAD_KINDS = frozenset(
    {FragmentKind.OVERHEAD_COMPACTION, FragmentKind.OVERHEAD_FACT_EXTRACTION}
)


class FragmentStatus(StrEnum):
    """Lifecycle status for a ``fragments`` row.

    ``IN_PROGRESS``
        Streaming response being written incrementally via
        ``ResponseWriter``.  Invisible to ``load_messages``.
    ``COMPLETE``
        Normal finished row.  Visible to ``load_messages``.
    ``FAILED``
        Failed LLM turn.  Visible in ``search_history``
        (user can retry via up-arrow) but excluded from
        ``load_messages`` (not part of replay history).
    """

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


# ── Incident enums ──────────────────────────────────────────────────────


class FailureKind(StrEnum):
    """Why an LLM attempt failed.

    Matches the error-classification branches in ``handle_llm``.
    """

    HISTORY_FORMAT = "history_format"
    """Provider rejected history structure (tool pairing, reasoning IDs, etc.)."""

    OVERFLOW = "overflow"
    """Context window exceeded."""

    EFFORT_UNSUPPORTED = "effort_unsupported"
    """Model does not support the ``effort`` parameter."""

    TOOL_ARGS = "tool_args"
    """``ValueError`` from malformed tool-call args."""

    TYPE_ERROR = "type_error"
    """``TypeError`` from adapter crash on unexpected null values."""

    CANCELLED = "cancelled"
    """User cancelled the turn (Ctrl+C)."""

    UNKNOWN = "unknown"
    """Unclassified failure."""


class RecoveryStrategy(StrEnum):
    """What transformation was applied to recover from a failure.

    Each value corresponds to a concrete code path in
    ``handle_llm`` or ``_prepare_turn``.
    """

    SIMPLIFY_HISTORY = "simplify_history"
    """Level 2: ``demote_thinking`` + ``flatten_tool_exchanges``, then retry."""

    COMPACT_THEN_RETRY = "compact_then_retry"
    """``compact_history``, then retry with compacted context."""

    EFFORT_OFF = "effort_off"
    """Disable ``effort`` parameter, then retry."""

    REPAIR_DANGLING = "repair_dangling"
    """Inject synthetic ``(cancelled)`` tool returns in memory."""

    DEMOTE_THINKING = "demote_thinking"
    """Convert ``ThinkingPart`` to ``TextPart`` with ``<thinking>`` tags."""

    CONSOLIDATE_TOOL_RETURNS = "consolidate_tool_returns"
    """Restructure tool returns so each response's returns are in one request."""

    FLATTEN_TOOL_EXCHANGES = "flatten_tool_exchanges"
    """Convert tool-call/result pairs to plain text (last resort)."""

    NONE = "none"
    """No recovery attempted (unrecoverable failure)."""


class IncidentOutcome(StrEnum):
    """Outcome of a recovery attempt."""

    RECOVERED = "recovered"
    FAILED = "failed"
    ABORTED = "aborted"


# ── TypeAdapters ─────────────────────────────────────────────────────
#
# _message_ta:  Serialises full ModelMessage (we strip ``parts``
#               after dumping for the header row, and re-attach
#               them from content rows when loading).
#
# _part_ta:     Serialises individual parts.  PydanticAI parts
#               already carry ``part_kind`` as a Literal field,
#               so the discriminated union just works.

_message_ta: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)

AnyPart = Annotated[
    SystemPromptPart
    | UserPromptPart
    | ToolReturnPart
    | RetryPromptPart
    | TextPart
    | ToolCallPart
    | BuiltinToolCallPart
    | BuiltinToolReturnPart
    | ThinkingPart
    | FilePart,
    Discriminator("part_kind"),
]

_part_ta: TypeAdapter[AnyPart] = TypeAdapter(AnyPart)


# ── Fragment ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Fragment:
    """All columns for a single ``fragments`` table row.

    Field order matches the SQL INSERT column order so
    ``dataclasses.astuple(row)`` produces the right positional args.
    """

    id: str
    session_id: str
    message_id: str | None
    fragment_index: int
    fragment_kind: FragmentKind
    created_at: str
    session_label: str | None
    repo_owner: str | None
    repo_name: str | None
    model_name: str | None
    review_target: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cost: float | None
    data_json: str | None
    user_text: str | None
    tool_name: str | None
    compacted_by: str | None
    status: FragmentStatus


# ── Incident models ─────────────────────────────────────────────
#
# Pydantic BaseModels for the ``data_json`` column of each incident
# ``FragmentKind``.  All optional fields default to ``None`` so
# callers only populate what is available.


class FailedAttempt(BaseModel, extra="ignore"):
    """``data_json`` for ``FragmentKind.LLM_ATTEMPT_FAILED``."""

    turn_id: str
    """Message ID of the failed ``REQUEST_MESSAGE``."""

    failure_kind: FailureKind
    """Why the attempt failed."""

    strategy: RecoveryStrategy
    """What will be attempted next (or ``NONE``)."""

    model_name: str | None = None
    """Full model identifier (e.g. ``claude/claude-sonnet-4-20250514``)."""

    status_code: int | None = None
    """HTTP status code from the provider, if ``ModelHTTPError``."""

    error_text: str | None = None
    """Normalised provider error message."""

    diagnostic: str | None = None
    """Full traceback or provider error body, verbatim."""

    history_message_count: int | None = None
    """Number of messages in history at time of failure."""

    estimated_context_tokens: int | None = None
    """Rough token estimate of the history sent."""

    effort_enabled: bool | None = None
    """Whether the ``effort`` parameter was active."""

    already_compacted: bool | None = None
    """Whether the session had been compacted before this attempt."""

    outcome: IncidentOutcome | None = None
    """Set after retry: ``recovered``, ``failed``, or ``aborted``."""


class HistoryRepair(BaseModel, extra="ignore"):
    """``data_json`` for ``FragmentKind.LLM_HISTORY_REPAIR``."""

    strategy: RecoveryStrategy
    """Which repair was applied."""

    turn_id: str | None = None
    """Message ID of the ``REQUEST_MESSAGE`` this repair belongs to."""

    reason: str | None = None
    """Why this repair was applied (e.g. ``cancelled_mid_tool_call``,
    ``cross_provider_retry``)."""

    # ── REPAIR_DANGLING detail ───────────────────────────────────

    tool_names: list[str] | None = None
    """Names of orphaned tools that were patched."""

    call_count: int | None = None
    """Number of tool calls patched."""

    # ── CONSOLIDATE_TOOL_RETURNS detail ────────────────────────────

    turns_fixed: int | None = None
    """Number of tool-call turns whose returns were restructured."""

    # ── DEMOTE_THINKING detail ───────────────────────────────────

    parts_demoted: int | None = None
    """Number of ``ThinkingPart``\\s converted to ``TextPart``."""

    # ── FLATTEN_TOOL_EXCHANGES detail ────────────────────────────

    tool_calls_flattened: int | None = None
    """Number of ``ToolCallPart``\\s converted to ``TextPart``."""

    tool_returns_flattened: int | None = None
    """Number of ``ToolReturnPart``\\s converted to ``UserPromptPart``."""

    retry_prompts_dropped: int | None = None
    """Number of ``RetryPromptPart``\\s removed."""


Incident = FailedAttempt | HistoryRepair
"""Union of all incident types for ``save_incident``."""


# ── Overhead models ──────────────────────────────────────────────
#
# Pydantic BaseModels for the ``data_json`` column of overhead
# fragments.  These track cost and metadata for background LLM
# calls (compaction summaries, fact extraction) that are not
# part of the conversation.


class CompactionTrigger(StrEnum):
    """What initiated the compaction."""

    MID_TURN = "mid-turn"
    AUTO_POST_TURN = "auto-post-turn"
    AUTO_OVERFLOW = "auto-overflow"
    MANUAL = "manual"


class CompactionOverhead(BaseModel):
    """``data_json`` for ``FragmentKind.OVERHEAD_COMPACTION``."""

    trigger: CompactionTrigger
    old_messages: int
    kept_messages: int
    summary_tokens: int
    model_name: str | None = None


class FactExtractionSource(StrEnum):
    """What triggered the fact extraction."""

    COMPACTION = "compaction"
    POST = "post"
    COMMAND = "command"


class FactExtractionOverhead(BaseModel):
    """``data_json`` for ``FragmentKind.OVERHEAD_FACT_EXTRACTION``."""

    source: FactExtractionSource
    added: int = 0
    confirmed: int = 0
    superseded: int = 0
    model_name: str | None = None
    fact_ids: list[str] = []


Overhead = CompactionOverhead | FactExtractionOverhead
"""Union of all overhead types for ``save_overhead``."""


# ═══════════════════════════════════════════════════════════════════════
# Write path: ModelMessage / incident / input → Fragment
# ═══════════════════════════════════════════════════════════════════════


def _dump_header(msg: ModelMessage) -> str:
    """Serialise message metadata to JSON, excluding parts.

    Uses PydanticAI's own TypeAdapter so complex fields
    (``RequestUsage``, ``FinishReason``, ``datetime``) are
    handled correctly.
    """
    data: dict[str, Any] = _message_ta.dump_python(msg, mode="json")
    data.pop("parts", None)
    return json.dumps(data)


def dump_part(part: ModelRequestPart | ModelResponsePart) -> str:
    """Serialise a single PydanticAI part to JSON."""
    return _part_ta.dump_json(part).decode()  # type: ignore[arg-type]  # AnyPart covers all concrete part types


def prepare_message_row(
    message: ModelMessage,
    *,
    context: SessionContext,
    row_id: str,
    status: FragmentStatus = FragmentStatus.COMPLETE,
) -> Fragment:
    """Build a message-level ``Fragment``.

    Serialises the message metadata (everything except ``parts``)
    into ``data_json``.
    """
    now = datetime.now(UTC).isoformat()

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    match message:
        case ModelRequest():
            kind = FragmentKind.REQUEST_MESSAGE
        case ModelResponse(usage=usage):
            kind = FragmentKind.RESPONSE_MESSAGE
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            cache_read_tokens = usage.cache_read_tokens or None
            cache_write_tokens = usage.cache_write_tokens or None
        case _:
            msg = f"Unsupported message type: {type(message)}"
            raise TypeError(msg)

    return Fragment(
        id=row_id,
        session_id=context.session_id,
        message_id=row_id,
        fragment_index=0,
        fragment_kind=kind,
        created_at=now,
        session_label=context.session_label,
        repo_owner=context.repo_owner,
        repo_name=context.repo_name,
        model_name=context.model_name,
        review_target=context.review_target,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cost=None,
        data_json=_dump_header(message),
        user_text=None,
        tool_name=None,
        compacted_by=None,
        status=status,
    )


def prepare_part_row(
    part: ModelRequestPart | ModelResponsePart,
    *,
    message_id: str,
    fragment_index: int,
    context: SessionContext,
    row_id: str,
    status: FragmentStatus = FragmentStatus.COMPLETE,
) -> Fragment:
    """Build a content ``Fragment`` for a single message part."""
    now = datetime.now(UTC).isoformat()
    fk = FragmentKind(part.part_kind)

    user_text: str | None = None
    tool_name: str | None = None

    if isinstance(part, UserPromptPart):
        user_text = part.content if isinstance(part.content, str) else str(part.content)
    elif isinstance(part, (ToolCallPart, ToolReturnPart, RetryPromptPart)):
        tool_name = part.tool_name

    return Fragment(
        id=row_id,
        session_id=context.session_id,
        message_id=message_id,
        fragment_index=fragment_index,
        fragment_kind=fk,
        created_at=now,
        session_label=context.session_label,
        repo_owner=context.repo_owner,
        repo_name=context.repo_name,
        model_name=context.model_name,
        review_target=context.review_target,
        input_tokens=None,
        output_tokens=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost=None,
        data_json=dump_part(part),
        user_text=user_text,
        tool_name=tool_name,
        compacted_by=None,
        status=status,
    )


def prepare_part_rows(
    message: ModelMessage,
    *,
    message_id: str,
    context: SessionContext,
    status: FragmentStatus = FragmentStatus.COMPLETE,
) -> list[Fragment]:
    """Build ``Fragment`` list for all parts in a message.

    Generates row IDs via UUID7.  Fragment index starts at 1
    (0 is the message-level row).
    """
    from uuid_utils import uuid7  # deferred: lightweight import

    rows: list[Fragment] = []
    match message:
        case ModelRequest(parts=parts) | ModelResponse(parts=parts):
            for i, part in enumerate(parts, start=1):
                rows.append(
                    prepare_part_row(
                        part,
                        message_id=message_id,
                        fragment_index=i,
                        context=context,
                        row_id=str(uuid7()),
                        status=status,
                    )
                )
    return rows


def prepare_incident_row(
    kind: FragmentKind,
    payload: Incident,
    *,
    context: SessionContext,
    row_id: str,
) -> Fragment:
    """Build a self-referencing ``Fragment`` for an incident.

    The payload is serialised via ``model_dump_json(exclude_none=True)``
    and stored in ``data_json``.
    """
    now = datetime.now(UTC).isoformat()
    return Fragment(
        id=row_id,
        session_id=context.session_id,
        message_id=row_id,
        fragment_index=0,
        fragment_kind=kind,
        created_at=now,
        session_label=context.session_label,
        repo_owner=context.repo_owner,
        repo_name=context.repo_name,
        model_name=context.model_name,
        review_target=context.review_target,
        input_tokens=None,
        output_tokens=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost=None,
        data_json=payload.model_dump_json(exclude_none=True),
        user_text=None,
        tool_name=None,
        compacted_by=None,
        status=FragmentStatus.COMPLETE,
    )


def prepare_overhead_row(
    kind: FragmentKind,
    payload: Overhead,
    *,
    context: SessionContext,
    row_id: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost: float | None = None,
) -> Fragment:
    """Build a self-referencing ``Fragment`` for overhead cost tracking.

    Same pattern as ``prepare_incident_row`` — self-referencing with
    ``message_id = id``, ``fragment_index = 0``.  Additionally
    populates ``input_tokens``, ``output_tokens``, and ``cost``
    columns from the overhead LLM call.
    """
    now = datetime.now(UTC).isoformat()
    return Fragment(
        id=row_id,
        session_id=context.session_id,
        message_id=row_id,
        fragment_index=0,
        fragment_kind=kind,
        created_at=now,
        session_label=context.session_label,
        repo_owner=context.repo_owner,
        repo_name=context.repo_name,
        model_name=context.model_name,
        review_target=context.review_target,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost=cost,
        data_json=payload.model_dump_json(exclude_none=True),
        user_text=None,
        tool_name=None,
        compacted_by=None,
        status=FragmentStatus.COMPLETE,
    )


def prepare_input_row(
    text: str,
    kind: FragmentKind,
    *,
    context: SessionContext,
    row_id: str,
) -> Fragment:
    """Build a self-referencing ``Fragment`` for a command/shell input.

    Sets ``user_text`` for search; no ``data_json``.
    """
    now = datetime.now(UTC).isoformat()
    return Fragment(
        id=row_id,
        session_id=context.session_id,
        message_id=row_id,
        fragment_index=0,
        fragment_kind=kind,
        created_at=now,
        session_label=context.session_label,
        repo_owner=context.repo_owner,
        repo_name=context.repo_name,
        model_name=context.model_name,
        review_target=context.review_target,
        input_tokens=None,
        output_tokens=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost=None,
        data_json=None,
        user_text=text,
        tool_name=None,
        compacted_by=None,
        status=FragmentStatus.COMPLETE,
    )


# ═══════════════════════════════════════════════════════════════════════
# Read path: Fragment → ModelMessage
# ═══════════════════════════════════════════════════════════════════════


def reconstruct_message(
    kind: FragmentKind,
    header_json: str,
    part_jsons: list[str],
) -> ModelMessage:
    """Rebuild a ``ModelMessage`` from its header JSON and part JSONs.

    Merges the header dict (message-level metadata) with the
    deserialised parts, then validates through PydanticAI's
    ``TypeAdapter[ModelMessage]``.

    After reconstruction, validates that all ``ToolCallPart`` args
    are parseable JSON.  Malformed args (e.g. from a model producing
    invalid JSON during streaming) are repaired to ``{}`` in-place
    so the message stays in the history and tool-call / tool-return
    pairing is preserved.
    """
    data: dict[str, Any] = json.loads(header_json)
    data["parts"] = [json.loads(pj) for pj in part_jsons]
    msg = _message_ta.validate_python(data)
    _validate_tool_call_args(msg)
    return msg


def _validate_tool_call_args(msg: ModelMessage) -> None:
    """Ensure every ``ToolCallPart`` has parseable args.

    PydanticAI accepts any string for ``ToolCallPart.args`` at
    validation time, but provider adapters call ``args_as_dict()``
    (which uses ``pydantic_core.from_json``) when building the
    API request.  If the model produced malformed JSON during
    streaming (e.g. mixed XML/JSON), the error surfaces far from
    the source.

    Rather than raising (which would skip the entire message and
    orphan its matching ``ToolReturnPart``\\s in subsequent
    ``ModelRequest``\\s), corrupt args are **repaired to ``{}``**
    in-place.  This preserves tool-call / tool-return pairing so
    the provider adapter never rejects the history on account of
    orphaned returns.
    """
    match msg:
        case ModelResponse(parts=parts):
            for part in parts:
                if isinstance(part, ToolCallPart):
                    try:
                        part.args_as_dict()
                    except ValueError:
                        log.warning(
                            "sessions: repairing corrupt args for tool call %s",
                            part.tool_name,
                        )
                        part.args = {}
