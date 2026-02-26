"""History repair and compaction helpers.

Includes cross-provider compatibility (``demote_thinking``) and
context-compaction utilities (``split_history``, ``serialise_for_summary``,
``build_summary_message``).
"""

from __future__ import annotations

import json
from dataclasses import replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def is_history_format_error(exc: Exception) -> bool:
    """Check if an API error is caused by malformed history items.

    PydanticAI stores provider-specific reasoning IDs in message history.
    When history is replayed to a different provider (or API variant),
    these IDs may be rejected (e.g. OpenAI Responses API expects ``rs_*``).

    Must be narrow — other 400 errors (orphaned tool returns, missing
    tool calls) are NOT format errors and should not trigger retry.
    """
    msg = str(exc).lower()
    # "Invalid 'input[6].id': 'reasoning_content'" — wrong ID format.
    # Match the specific pattern: "invalid" + "'input[" or ".id'"
    if "invalid" in msg and ("'input[" in msg or ".id'" in msg):
        return True
    # "Item 'fc_...' was provided without its required 'reasoning' item"
    return "provided without" in msg and "reasoning" in msg


def demote_thinking(history: list[ModelMessage]) -> list[ModelMessage]:
    """Return history with ThinkingParts converted to plain TextParts.

    Wraps thinking content in ``<thinking>`` tags so the model can
    still see prior reasoning, without the provider-specific IDs
    that cause cross-provider errors.
    """
    cleaned: list[ModelMessage] = []
    for msg in history:
        if isinstance(msg, ModelResponse):
            parts = [
                TextPart(content=f"<thinking>\n{p.content}\n</thinking>")
                if isinstance(p, ThinkingPart) and p.content
                else p
                for p in msg.parts
                if not isinstance(p, ThinkingPart) or p.content
            ]
            if parts:
                cleaned.append(replace(msg, parts=parts))
        else:
            cleaned.append(msg)
    return cleaned


# ── Compaction helpers ───────────────────────────────────────────────

_SUMMARY_MARKER = "[Context summary — earlier conversation was compacted]"


def _is_user_turn_start(msg: ModelMessage) -> bool:
    """Check if a message starts a new user turn.

    A turn begins with a ``ModelRequest`` that contains at least one
    ``UserPromptPart`` (as opposed to tool-return-only requests).
    """
    if not isinstance(msg, ModelRequest):
        return False
    return any(isinstance(p, UserPromptPart) for p in msg.parts)


def split_history(
    history: list[ModelMessage],
    keep_turns: int,
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    """Split history into (old, kept) by user turn count.

    A *turn* starts at a ``ModelRequest`` containing a ``UserPromptPart``
    and includes all subsequent messages until the next such request.

    Returns ``(old, kept)`` where ``kept`` has the last *keep_turns*
    turns and ``old`` has everything before.  If the history has
    ``keep_turns`` or fewer turns, ``old`` is empty.

    After splitting, any tool-return-only ``ModelRequest`` messages in
    ``kept`` whose ``tool_call_id`` has no matching ``ToolCallPart``
    in ``kept`` are moved to ``old`` to prevent orphaned tool returns.
    """
    # Find indices where each user turn starts.
    turn_starts: list[int] = [i for i, msg in enumerate(history) if _is_user_turn_start(msg)]

    if len(turn_starts) <= keep_turns:
        return [], list(history)

    # The cut point is where the kept turns begin.
    cut = turn_starts[-keep_turns]
    old, kept = list(history[:cut]), list(history[cut:])

    # Move orphaned tool-return requests into old.
    kept_call_ids = _collect_tool_call_ids(kept)
    clean: list[ModelMessage] = []
    for msg in kept:
        if _is_orphaned_tool_return(msg, kept_call_ids):
            old.append(msg)
        else:
            clean.append(msg)

    return old, clean


def _collect_tool_call_ids(messages: list[ModelMessage]) -> set[str]:
    """Collect all tool_call_id values from ToolCallParts in responses."""
    ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id:
                    ids.add(part.tool_call_id)
    return ids


def _is_orphaned_tool_return(msg: ModelMessage, known_call_ids: set[str]) -> bool:
    """Check if a message is a tool-return-only request with no matching call."""
    if not isinstance(msg, ModelRequest):
        return False
    has_user_prompt = any(isinstance(p, UserPromptPart) for p in msg.parts)
    if has_user_prompt:
        return False
    tool_returns = [
        p for p in msg.parts if isinstance(p, ToolReturnPart) and p.tool_call_id is not None
    ]
    if not tool_returns:
        return False
    return all(p.tool_call_id not in known_call_ids for p in tool_returns)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def serialise_for_summary(
    messages: list[ModelMessage],
    max_tool_chars: int = 2_000,
) -> str:
    """Convert messages to a human-readable text for the summary prompt.

    Tool return content is truncated to *max_tool_chars* to keep the
    serialised output manageable.  Thinking parts are omitted.
    """
    sections: list[str] = []

    for msg in messages:
        match msg:
            case ModelRequest(parts=req_parts):
                for req_part in req_parts:
                    match req_part:
                        case UserPromptPart(content=content):
                            text = content if isinstance(content, str) else str(content)
                            sections.append(f"## User\n{text}")
                        case ToolReturnPart(tool_name=name, content=content):
                            text = str(content)
                            if len(text) > max_tool_chars:
                                text = text[:max_tool_chars] + "…[truncated]"
                            sections.append(f"## Tool result: {name}\n{text}")
            case ModelResponse(parts=resp_parts):
                for resp_part in resp_parts:
                    match resp_part:
                        case TextPart(content=content):
                            sections.append(f"## Assistant\n{content}")
                        case ToolCallPart(tool_name=name, args=args):
                            if isinstance(args, dict):
                                args_str = json.dumps(args, ensure_ascii=False)
                            elif args is not None:
                                args_str = str(args)
                            else:
                                args_str = ""
                            sections.append(f"## Tool call: {name}({args_str})")
                        # ThinkingPart — omit from summary input

    return "\n\n".join(sections)


def build_summary_message(summary_text: str) -> ModelRequest:
    """Create a synthetic first message containing the compaction summary."""
    content = f"{_SUMMARY_MARKER}\n\n{summary_text}"
    return ModelRequest(parts=[UserPromptPart(content=content)])
