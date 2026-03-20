"""Session data types — kinds, status, context, and result types.

These are the core types that define the schema's type system.
Every other sessions module depends on these. Pure data types
with no heavy dependencies (no `pydantic_ai`, `duckdb`, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

GLOBAL_SCOPE = "global"
"""Scope value for facts that apply across all repositories."""


class FragmentKind(StrEnum):
    """Row kind for the `fragments` table.

    Five groups of values share a single TEXT column:

    - **Message-level**: `REQUEST_MESSAGE`, `RESPONSE_MESSAGE`
    - **PydanticAI parts**: values match `part_kind` strings so
      `FragmentKind(part.part_kind)` works for any content part
    - **User input**: `COMMAND`, `SHELL`
    - **Incidents**: `LLM_ATTEMPT_FAILED`, `LLM_HISTORY_REPAIR`
    - **Overhead**: `OVERHEAD_COMPACTION`, `OVERHEAD_FACT_EXTRACTION`

    Use `is_message`, `is_input`, `is_incident`,
    `is_overhead` to test group membership.
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
    """Lifecycle status for a `fragments` row.

    `IN_PROGRESS`
        Streaming response being written incrementally via
        `ResponseWriter`.  Invisible to `load_messages`.
    `COMPLETE`
        Normal finished row.  Visible to `load_messages`.
    `FAILED`
        Failed LLM turn.  Visible in `search_history`
        (user can retry via up-arrow) but excluded from
        `load_messages` (not part of replay history).
    """

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Caller-supplied metadata written onto every row."""

    session_id: str
    session_label: str | None = None
    repo_owner: str | None = None
    repo_name: str | None = None
    model_name: str | None = None
    review_target: str | None = None


@dataclass(frozen=True, slots=True)
class Fragment:
    """All columns for a single `fragments` table row.

    Field order matches the SQL INSERT column order so
    `dataclasses.astuple(row)` produces the right positional args.
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


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Lightweight session listing — no message bodies."""

    session_id: str
    session_label: str | None
    last_active: str
    message_count: int
    total_cost: float
    model_name: str | None
    review_target: str | None
    repo_owner: str | None
    repo_name: str | None


@dataclass(frozen=True, slots=True)
class Fact:
    """A single cross-session memory fact."""

    id: str
    scope: str
    content: str
    source_session_id: str
    created_at: str
    last_confirmed_at: str
    confirm_count: int
    superseded_by: str | None = None
