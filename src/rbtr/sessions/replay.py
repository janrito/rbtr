"""Replay stored conversation history as display events.

Converts a `list[ModelMessage]` into the same events the live
streaming path produces — `InputEcho`, `MarkdownOutput`,
`FlushPanel`, `ToolCallStarted`, `ToolCallFinished`.  The TUI
renders replayed history through its existing event handlers,
so there is a single rendering route for both live and replay.

Public API
----------
- `replay_history` — walk messages, emit events via a callback.
- `format_tool_args` — serialise tool-call args to a display string.
- `is_compaction_summary` — detect compaction summary messages.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import assert_never

from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from rbtr.events import (
    Event,
    FlushPanel,
    InputEcho,
    MarkdownOutput,
    Output,
    PanelVariant,
    ToolCallFinished,
    ToolCallStarted,
)
from rbtr.sessions.kinds import SUMMARY_MARKER

_COMPACTION_PREVIEW_CHARS = 200


# ── Public API ───────────────────────────────────────────────────────


def format_tool_args(args: object) -> str:
    """Serialise `ToolCallPart.args` to a display string.

    `dict` → compact JSON, other truthy values → `str()`, `None` → `""`.
    """
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    if args is not None:
        return str(args)
    return ""


def is_compaction_summary(text: str) -> bool:
    """Return whether *text* is a compaction summary message."""
    return text.startswith(SUMMARY_MARKER)


def replay_history(
    emit: Callable[[Event], None],
    messages: list[ModelMessage],
) -> None:
    """Replay conversation history by emitting display events.

    Walks the message list and emits the same events the live path
    would produce so the TUI renders history through a single route.

    Uses exhaustive ``match`` with ``assert_never`` on both
    ``ModelRequestPart`` and ``ModelResponsePart`` unions so that
    mypy catches any unhandled part type.
    """
    # Pass 1: collect tool results by call ID.
    tool_results: dict[str, tuple[str, bool]] = {}
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            continue
        for req_part in msg.parts:
            match req_part:
                case ToolReturnPart(tool_call_id=cid, content=content):
                    tool_results[cid] = (str(content), False)
                case RetryPromptPart(tool_call_id=cid, content=content):
                    tool_results[cid] = (str(content), True)
                case UserPromptPart() | SystemPromptPart():
                    pass
                case _:
                    assert_never(req_part)

    # Pass 2: walk messages and emit events.
    for msg in messages:
        if isinstance(msg, ModelRequest):
            _replay_request(emit, msg)
        elif isinstance(msg, ModelResponse):
            _replay_response(emit, msg, tool_results)


# ── Internal helpers ─────────────────────────────────────────────────


def _replay_request(
    emit: Callable[[Event], None],
    msg: ModelRequest,
) -> None:
    """Emit events for a single ``ModelRequest``."""
    for part in msg.parts:
        match part:
            case UserPromptPart(content=content):
                text = str(content)
                if is_compaction_summary(text):
                    preview = text[:_COMPACTION_PREVIEW_CHARS]
                    if len(text) > _COMPACTION_PREVIEW_CHARS:
                        preview += "…"
                    emit(Output(text=preview))
                    emit(FlushPanel(variant=PanelVariant.QUEUED))
                else:
                    emit(InputEcho(text=text))
            case SystemPromptPart() | ToolReturnPart() | RetryPromptPart():
                pass
            case _:
                assert_never(part)


def _replay_response(
    emit: Callable[[Event], None],
    msg: ModelResponse,
    tool_results: dict[str, tuple[str, bool]],
) -> None:
    """Emit events for a single ``ModelResponse``."""
    accumulated_text = ""
    for part in msg.parts:
        match part:
            case TextPart(content=content):
                accumulated_text += content
            case ToolCallPart(tool_name=name, args=args, tool_call_id=cid):
                if accumulated_text:
                    emit(MarkdownOutput(text=accumulated_text))
                    emit(FlushPanel(variant=PanelVariant.RESPONSE))
                    accumulated_text = ""
                emit(
                    ToolCallStarted(
                        tool_name=name,
                        args=format_tool_args(args),
                        tool_call_id=cid,
                    )
                )
                result_text, failed = tool_results.get(cid, ("", False))
                emit(
                    ToolCallFinished(
                        tool_name=name,
                        tool_call_id=cid,
                        result="" if failed else result_text,
                        error=result_text if failed else None,
                    )
                )
            case ThinkingPart() | BuiltinToolCallPart() | BuiltinToolReturnPart() | FilePart():
                pass
            case _:
                assert_never(part)

    if accumulated_text:
        emit(MarkdownOutput(text=accumulated_text))
        emit(FlushPanel(variant=PanelVariant.RESPONSE))
