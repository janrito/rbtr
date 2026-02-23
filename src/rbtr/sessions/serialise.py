"""Serialisation helpers for session persistence.

Single entry point: ``prepare_row()`` accepts either a
``ModelMessage`` (for LLM request/response rows) or a plain
``str`` (for ``/command`` and ``!shell`` rows).  Identity checks
route to the appropriate extraction logic.

All functions are pure — no I/O, no engine or UI imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

_message_ta: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)

# ── Types ────────────────────────────────────────────────────────────


class MessageKind(StrEnum):
    """Row kind — determines how the row is interpreted."""

    REQUEST = "request"
    RESPONSE = "response"
    COMMAND = "command"
    SHELL = "shell"
    COMPACTION = "compaction"


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Caller-supplied context written onto every row."""

    session_id: str
    session_label: str | None = None
    repo_owner: str | None = None
    repo_name: str | None = None
    model_name: str | None = None


@dataclass(frozen=True, slots=True)
class MessageRow:
    """All columns for a single ``messages`` table row.

    Returned by ``prepare_row()`` — the store inserts this directly.
    """

    id: str
    session_id: str
    created_at: str
    session_label: str | None
    repo_owner: str | None
    repo_name: str | None
    model_name: str | None
    kind: MessageKind
    message_json: str | None
    user_text: str | None
    tool_names: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost: float | None
    compacted_by: str | None


# ── Public API ───────────────────────────────────────────────────────


def prepare_row(
    message: ModelMessage | str,
    *,
    context: SessionContext,
    row_id: str,
    kind: MessageKind | None = None,
    cost: float | None = None,
) -> MessageRow:
    """Build a ``MessageRow`` from a message and caller context.

    Parameters
    ----------
    message:
        A ``ModelRequest`` or ``ModelResponse`` for LLM messages,
        or a plain ``str`` for ``/command`` and ``!shell`` input.
    context:
        Session-level metadata written onto the row.
    row_id:
        Pre-generated UUID7 for the row's primary key.
    kind:
        Explicit kind override — required for ``str`` messages
        (``'command'`` or ``'shell'``).  Ignored for ``ModelMessage``
        (derived from ``message.kind``).
    cost:
        USD cost of this turn (only meaningful for response rows).
    """
    now = datetime.now(UTC).isoformat()

    match message:
        case ModelRequest() | ModelResponse():
            return _from_model_message(
                message,
                context=context,
                row_id=row_id,
                now=now,
                cost=cost,
            )
        case str():
            if kind is None:
                msg = "kind is required for str messages ('command' or 'shell')"
                raise ValueError(msg)
            return _from_text(message, kind=kind, context=context, row_id=row_id, now=now)
        case _:
            msg = f"Unsupported message type: {type(message)}"
            raise TypeError(msg)


def serialise_message(message: ModelMessage) -> str:
    """Serialise a single ``ModelMessage`` to JSON."""
    return _message_ta.dump_json(message).decode()


def deserialise_message(json_str: str) -> ModelMessage:
    """Deserialise a JSON string back to a ``ModelMessage``."""
    return _message_ta.validate_json(json_str)


# ── Internal helpers ─────────────────────────────────────────────────


def _from_model_message(
    message: ModelMessage,
    *,
    context: SessionContext,
    row_id: str,
    now: str,
    cost: float | None,
) -> MessageRow:
    """Extract a row from a PydanticAI ``ModelMessage``."""
    message_json = serialise_message(message)
    msg_kind = MessageKind(message.kind)  # 'request' | 'response'

    user_text: str | None = None
    tool_names: list[str] = []
    input_tokens: int | None = None
    output_tokens: int | None = None

    match message:
        case ModelRequest(parts=req_parts):
            for req_part in req_parts:
                if user_text is None:
                    user_text = _user_text_from_part(req_part)
                if isinstance(req_part, ToolReturnPart):
                    tool_names.append(req_part.tool_name)

        case ModelResponse(parts=resp_parts, usage=usage):
            for resp_part in resp_parts:
                if isinstance(resp_part, ToolCallPart):
                    tool_names.append(resp_part.tool_name)
            if usage is not None:
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens

    return MessageRow(
        id=row_id,
        session_id=context.session_id,
        created_at=now,
        session_label=context.session_label,
        repo_owner=context.repo_owner,
        repo_name=context.repo_name,
        model_name=context.model_name,
        kind=msg_kind,
        message_json=message_json,
        user_text=user_text,
        tool_names=",".join(tool_names) if tool_names else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        compacted_by=None,
    )


def _from_text(
    text: str,
    *,
    kind: MessageKind,
    context: SessionContext,
    row_id: str,
    now: str,
) -> MessageRow:
    """Build a row for a ``/command`` or ``!shell`` input."""
    return MessageRow(
        id=row_id,
        session_id=context.session_id,
        created_at=now,
        session_label=context.session_label,
        repo_owner=context.repo_owner,
        repo_name=context.repo_name,
        model_name=context.model_name,
        kind=kind,
        message_json=None,
        user_text=text,
        tool_names=None,
        input_tokens=None,
        output_tokens=None,
        cost=None,
        compacted_by=None,
    )


def _user_text_from_part(part: ModelRequestPart) -> str | None:
    """Extract user text from a request part, if applicable."""
    if isinstance(part, UserPromptPart):
        return part.content if isinstance(part.content, str) else str(part.content)
    return None
