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
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import Discriminator, TypeAdapter
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

# ── Types ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Caller-supplied metadata written onto every row."""

    session_id: str
    session_label: str | None = None
    repo_owner: str | None = None
    repo_name: str | None = None
    model_name: str | None = None


class FragmentKind(StrEnum):
    """Row kind for the ``fragments`` table.

    Values match PydanticAI ``part_kind`` strings exactly so
    ``FragmentKind(part.part_kind)`` works for any content part.
    """

    REQUEST_MESSAGE = "request-message"
    RESPONSE_MESSAGE = "response-message"
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
    COMMAND = "command"
    SHELL = "shell"


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
    input_tokens: int | None
    output_tokens: int | None
    cost: float | None
    data_json: str | None
    user_text: str | None
    tool_name: str | None
    compacted_by: str | None
    complete: int


# ═══════════════════════════════════════════════════════════════════════
# Write path: ModelMessage → Fragment
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


def _dump_part(part: ModelRequestPart | ModelResponsePart) -> str:
    """Serialise a single PydanticAI part to JSON."""
    return _part_ta.dump_json(part).decode()  # type: ignore[arg-type]  # AnyPart covers all concrete part types


def prepare_message_row(
    message: ModelMessage,
    *,
    context: SessionContext,
    row_id: str,
    complete: bool = True,
) -> Fragment:
    """Build a message-level ``Fragment``.

    Serialises the message metadata (everything except ``parts``)
    into ``data_json``.
    """
    now = datetime.now(UTC).isoformat()

    input_tokens: int | None = None
    output_tokens: int | None = None

    match message:
        case ModelRequest():
            kind = FragmentKind.REQUEST_MESSAGE
        case ModelResponse(usage=usage):
            kind = FragmentKind.RESPONSE_MESSAGE
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
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
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=None,
        data_json=_dump_header(message),
        user_text=None,
        tool_name=None,
        compacted_by=None,
        complete=1 if complete else 0,
    )


def prepare_part_row(
    part: ModelRequestPart | ModelResponsePart,
    *,
    message_id: str,
    fragment_index: int,
    context: SessionContext,
    row_id: str,
    complete: bool = True,
) -> Fragment:
    """Build a content ``Fragment`` for a single message part."""
    now = datetime.now(UTC).isoformat()
    fk = FragmentKind(part.part_kind)

    user_text: str | None = None
    tool_name: str | None = None

    if isinstance(part, UserPromptPart):
        user_text = part.content if isinstance(part.content, str) else str(part.content)
    elif isinstance(part, (ToolCallPart, ToolReturnPart)):
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
        input_tokens=None,
        output_tokens=None,
        cost=None,
        data_json=_dump_part(part),
        user_text=user_text,
        tool_name=tool_name,
        compacted_by=None,
        complete=1 if complete else 0,
    )


def prepare_part_rows(
    message: ModelMessage,
    *,
    message_id: str,
    context: SessionContext,
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
                    )
                )
    return rows


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
    """
    data: dict[str, Any] = json.loads(header_json)
    data["parts"] = [json.loads(pj) for pj in part_jsons]
    return _message_ta.validate_python(data)
