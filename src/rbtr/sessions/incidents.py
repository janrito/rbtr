"""Incident data models for the ``data_json`` column.

LLM failure and history-repair records are persisted as
``Fragment`` rows with ``FragmentKind.LLM_ATTEMPT_FAILED`` or
``LLM_HISTORY_REPAIR``.  Their ``data_json`` column contains a
Pydantic ``BaseModel`` payload serialised via
``model_dump_json(exclude_none=True)``.  The models use
``extra="ignore"`` so payloads can evolve without migrations.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


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
