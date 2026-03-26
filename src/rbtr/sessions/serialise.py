"""Serialisation for session persistence.

Translates between PydanticAI message types and `Fragment`
objects for the `fragments` table.

Design
------
Each `ModelMessage` produces **1 + N** rows:

- **1 message row** (`message_id = id`, `fragment_index = 0`):
  stores message-level metadata (everything except `parts`).
- **N content rows** (one per part, `fragment_index` ≥ 1):
  stores each PydanticAI part as-is via its own `TypeAdapter`.

No custom intermediate models.  Message headers are serialised via
PydanticAI's `TypeAdapter[ModelMessage]` (parts stripped).  Parts
are serialised directly via their built-in `part_kind` discriminator.

All functions are pure — no I/O, no engine or UI imports.

"""

import json
import logging
from datetime import UTC, datetime
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

from rbtr.sessions.incidents import Incident
from rbtr.sessions.kinds import (
    Fragment,
    FragmentKind,
    FragmentStatus,
    SessionContext,
)
from rbtr.sessions.overhead import Overhead

log = logging.getLogger(__name__)


# ── TypeAdapters ─────────────────────────────────────────────────────
#
# _message_ta:  Serialises full ModelMessage (we strip `parts`
#               after dumping for the header row, and re-attach
#               them from content rows when loading).
#
# _part_ta:     Serialises individual parts.  PydanticAI parts
#               already carry `part_kind` as a Literal field,
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


# ═══════════════════════════════════════════════════════════════════════
# Write path: ModelMessage / incident / input → Fragment
# ═══════════════════════════════════════════════════════════════════════


def dump_header(msg: ModelMessage) -> str:
    """Serialise message metadata to JSON, excluding parts.

    Uses PydanticAI's own TypeAdapter so complex fields
    (`RequestUsage`, `FinishReason`, `datetime`) are
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
    """Build a message-level `Fragment`.

    Serialises the message metadata (everything except `parts`)
    into `data_json`.
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
        data_json=dump_header(message),
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
    """Build a content `Fragment` for a single message part."""
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
    """Build `Fragment` list for all parts in a message.

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
    """Build a self-referencing `Fragment` for an incident.

    The payload is serialised via `model_dump_json(exclude_none=True)`
    and stored in `data_json`.
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
    """Build a self-referencing `Fragment` for overhead cost tracking.

    Same pattern as `prepare_incident_row` — self-referencing with
    `message_id = id`, `fragment_index = 0`.  Additionally
    populates `input_tokens`, `output_tokens`, and `cost`
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
    """Build a self-referencing `Fragment` for a command/shell input.

    Sets `user_text` for search; no `data_json`.
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
    """Rebuild a `ModelMessage` from its header JSON and part JSONs.

    Merges the header dict (message-level metadata) with the
    deserialised parts, then validates through PydanticAI's
    `TypeAdapter[ModelMessage]`.
    """
    data: dict[str, Any] = json.loads(header_json)
    data["parts"] = [json.loads(pj) for pj in part_jsons]
    return _message_ta.validate_python(data)
