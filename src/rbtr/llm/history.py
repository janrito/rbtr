"""History repair and compaction helpers.

Cross-provider compatibility (``demote_thinking``,
``flatten_tool_exchanges``), structural repair
(``repair_dangling_tool_calls``), and context-compaction
utilities (``split_history``, ``serialise_for_summary``,
``build_summary_message``).

History rewrites
~~~~~~~~~~~~~~~~

LLM APIs require every ``ToolCallPart`` to have a matching
``ToolReturnPart`` (same ``tool_call_id``) and vice versa, and
require ``ToolCallPart.args`` to be valid JSON.  Providers also
encode tool exchanges in incompatible wire formats and reject
history containing metadata from other providers.

Seven mechanisms enforce these invariants at three stages.  See
the *History repair* section in the README for user-facing
documentation.

On load (before the first API call):

- ``_validate_tool_call_args`` (``serialise.py``) -- sets
  unparseable ``ToolCallPart.args`` to ``{}``.
- ``repair_dangling_tool_calls`` -- injects synthetic
  ``(cancelled)`` results for unmatched tool calls.

On retry (after a provider rejection):

- ``demote_thinking`` -- converts ``ThinkingPart`` to
  ``TextPart`` wrapped in ``<thinking>`` tags.
- ``flatten_tool_exchanges`` -- converts ``ToolCallPart`` /
  ``ToolReturnPart`` to plain text, removing structural pairing.

On compaction (when splitting history):

- ``split_history`` -- moves orphaned tool returns into *old*.
- ``snap_to_safe_boundary`` -- adjusts split point to avoid
  separating tool-call / tool-result pairs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

log = logging.getLogger(__name__)


# ── Result types ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DemoteResult:
    """Result of ``demote_thinking``."""

    history: list[ModelMessage]
    parts_demoted: int


@dataclass(frozen=True, slots=True)
class FlattenResult:
    """Result of ``flatten_tool_exchanges``."""

    history: list[ModelMessage]
    tool_calls_flattened: int
    tool_returns_flattened: int
    retry_prompts_dropped: int


def is_history_format_error(exc: Exception) -> bool:
    """Check if an API error is caused by malformed history items.

    Providers reject history for two broad reasons:

    * **Provider-specific metadata** -- reasoning IDs, thinking
      part formats, etc. that another provider doesn't recognise.
    * **Tool-call structure** -- pairing, ordering, or required
      fields that differ between wire formats.

    The patterns below are intentionally general (``tool_use`` +
    ``tool_result``, ``required`` + ``field``, etc.) rather than
    matching specific model error strings, so they cover
    OpenAI-compatible endpoints that relay upstream errors in
    varying formats.
    """
    msg = str(exc).lower()
    # Reasoning ID rejected: "'input[6].id': 'reasoning_content'"
    if "invalid" in msg and ("'input[" in msg or ".id'" in msg):
        return True
    # Missing reasoning item: "provided without its required 'reasoning' item"
    if "provided without" in msg and "reasoning" in msg:
        return True
    # Unpaired tool calls: "tool_use ids ... without tool_result blocks"
    if "tool_use" in msg and "tool_result" in msg:
        return True
    # Orphaned tool return: "No tool call found for function call output"
    if "no tool call found" in msg and "function call output" in msg:
        return True
    # Extra provider-specific fields rejected by a different provider
    # (e.g. OpenAI Responses API reasoning IDs sent to Fireworks):
    # "Extra inputs are not permitted, field: 'messages[1].rs_...'"
    if "extra inputs are not permitted" in msg:
        return True
    # Required field missing from a message (e.g. null content on an
    # assistant message that contains only tool calls).
    return "required" in msg and "field" in msg


def demote_thinking(history: list[ModelMessage]) -> DemoteResult:
    """Return history with ThinkingParts converted to plain TextParts.

    Wraps thinking content in ``<thinking>`` tags so the model can
    still see prior reasoning, without the provider-specific IDs
    that cause cross-provider errors.
    """
    cleaned: list[ModelMessage] = []
    parts_demoted = 0
    for msg in history:
        if isinstance(msg, ModelResponse):
            new_parts: list[ModelResponsePart] = []
            for p in msg.parts:
                if isinstance(p, ThinkingPart):
                    if p.content:
                        new_parts.append(TextPart(content=f"<thinking>\n{p.content}\n</thinking>"))
                        parts_demoted += 1
                    # Empty thinking parts are dropped (no increment).
                else:
                    new_parts.append(p)
            if new_parts:
                cleaned.append(replace(msg, parts=new_parts))
        else:
            cleaned.append(msg)
    return DemoteResult(history=cleaned, parts_demoted=parts_demoted)


def flatten_tool_exchanges(history: list[ModelMessage]) -> FlattenResult:
    """Convert tool-call exchanges into plain text.

    Cross-provider last resort
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Each provider encodes tool-call / tool-result exchanges in its own
    wire format (Anthropic ``tool_use`` / ``tool_result`` blocks, OpenAI
    ``function_call`` items, etc.).  PydanticAI translates between them,
    but edge cases in complex histories (multi-turn tool-call chains,
    compaction boundaries, partial streaming saves) can produce
    structures that the *target* provider's adapter rejects — even
    though ``repair_dangling_tool_calls`` already ensured every
    ``ToolCallPart`` has a matching ``ToolReturnPart`` in the internal
    representation.

    This function eliminates the structural pairing entirely by
    converting the parts to plain text:

    * ``ToolCallPart``  -> ``TextPart("[Repaired historical tool call -- tool_name(args)]")``
    * ``ToolReturnPart`` -> ``UserPromptPart("[Repaired historical tool result -- tool_name]\\n...")``
    * ``RetryPromptPart`` — dropped (only meaningful to the original
      provider).

    All content survives — file contents, grep output, diff patches
    etc. remain in the history as readable text.  Only the machine-
    level call/result pairing is removed.

    When it runs
    ~~~~~~~~~~~~

    Never on the first attempt.  The normal path is:

    1. ``repair_dangling_tool_calls`` fixes orphaned tool calls
       (Ctrl+C mid-turn).
    2. PydanticAI's ``_clean_message_history`` merges consecutive
       same-role messages so the provider adapter sees alternating
       ``user`` / ``assistant`` turns.
    3. The provider adapter converts the internal messages.

    If step 3 still produces a 400 error that
    ``is_history_format_error`` matches, ``handle_llm`` retries once
    with ``simplify_history=True``, which calls *this* function
    (alongside ``demote_thinking``).
    """
    cleaned: list[ModelMessage] = []
    tool_calls_flattened = 0
    tool_returns_flattened = 0
    retry_prompts_dropped = 0
    for msg in history:
        if isinstance(msg, ModelResponse):
            resp_parts: list[ModelResponsePart] = []
            for rp in msg.parts:
                if isinstance(rp, ToolCallPart):
                    tool_calls_flattened += 1
                    resp_parts.append(_flatten_response_part(rp))
                else:
                    resp_parts.append(rp)
            cleaned.append(ModelResponse(parts=resp_parts, model_name=msg.model_name))
        elif isinstance(msg, ModelRequest):
            for qp in msg.parts:
                if isinstance(qp, ToolReturnPart):
                    tool_returns_flattened += 1
                elif isinstance(qp, RetryPromptPart):
                    retry_prompts_dropped += 1
            req_parts = _flatten_request_parts(msg.parts)
            if req_parts:
                cleaned.append(ModelRequest(parts=req_parts))
        else:
            cleaned.append(msg)
    return FlattenResult(
        history=cleaned,
        tool_calls_flattened=tool_calls_flattened,
        tool_returns_flattened=tool_returns_flattened,
        retry_prompts_dropped=retry_prompts_dropped,
    )


def _flatten_response_part(part: ModelResponsePart) -> ModelResponsePart:
    """Convert a single response part for ``flatten_tool_exchanges``.

    ``ToolCallPart`` → ``TextPart`` with a bracketed summary that
    preserves the tool name and arguments.  All other part types
    (``TextPart``, ``ThinkingPart``, etc.) pass through unchanged.
    """
    if not isinstance(part, ToolCallPart):
        return part
    args = part.args
    if isinstance(args, dict):
        args_str = json.dumps(args, ensure_ascii=False)
    elif args is not None:
        args_str = str(args)
    else:
        args_str = ""
    return TextPart(content=f"[Repaired historical tool call -- {part.tool_name}({args_str})]")


def _flatten_request_parts(
    parts: Sequence[ModelRequestPart],
) -> list[ModelRequestPart]:
    """Convert request parts for ``flatten_tool_exchanges``.

    ``ToolReturnPart`` → ``UserPromptPart`` with the tool output
    prefixed by the tool name.  ``RetryPromptPart`` is dropped
    (retries are provider-internal and meaningless after a model
    switch).  All other parts (``UserPromptPart``,
    ``SystemPromptPart``) pass through unchanged.
    """
    out: list[ModelRequestPart] = []
    for p in parts:
        if isinstance(p, ToolReturnPart):
            text = f"[Repaired historical tool result -- {p.tool_name}]\n{p.content}"
            out.append(UserPromptPart(content=text))
        elif isinstance(p, RetryPromptPart):
            continue
        else:
            out.append(p)
    return out


# ── Dangling tool-call repair ────────────────────────────────────────


def repair_dangling_tool_calls(
    history: list[ModelMessage],
) -> tuple[list[ModelMessage], list[str], list[ModelMessage]]:
    """Fix history left dirty by a cancelled tool-calling turn.

    Part of the tool-call pairing invariant (see module docstring).
    For each ``ModelResponse`` with ``ToolCallPart``\\s, checks whether
    every call has a matching ``ToolReturnPart`` *anywhere* in the
    history.  Injects a synthetic ``(cancelled)`` result for any
    truly unmatched call.

    Called once on session load — not during normal operation.

    Returns ``(repaired_history, tool_names, new_messages)`` where
    *tool_names* lists every tool that was patched (empty if no
    repair needed), and *new_messages* contains only the synthetic
    ``ModelRequest``\\s that were injected (for persistence).
    """
    if not history:
        return history, [], []

    # First pass: collect all tool_call_ids that already have a
    # ToolReturnPart or RetryPromptPart anywhere in the history.
    # This prevents false positives when a user prompt is
    # interleaved between tool calls and their returns.
    globally_answered: set[str | None] = set()
    for msg in history:
        if isinstance(msg, ModelRequest):
            for p in msg.parts:
                if isinstance(p, (ToolReturnPart, RetryPromptPart)):
                    globally_answered.add(p.tool_call_id)

    repaired: list[ModelMessage] | None = None  # lazy copy
    all_tool_names: list[str] = []
    new_messages: list[ModelMessage] = []

    i = 0
    while i < len(history):
        msg = history[i]
        if isinstance(msg, ModelResponse):
            tool_calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
            if tool_calls:
                missing = [tc for tc in tool_calls if tc.tool_call_id not in globally_answered]
                if missing:
                    if repaired is None:
                        repaired = list(history[:i])
                    repaired.append(msg)

                    # Check if the immediately next message has partial
                    # results — keep it as-is if so.
                    next_msg = history[i + 1] if i + 1 < len(history) else None
                    if isinstance(next_msg, ModelRequest) and any(
                        isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in next_msg.parts
                    ):
                        repaired.append(next_msg)
                        i += 1  # skip next_msg, already added

                    names = [tc.tool_name for tc in missing]
                    all_tool_names.extend(names)
                    log.info(
                        "Repairing %d dangling tool call(s) from cancelled turn.",
                        len(missing),
                    )
                    synthetic = ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name=tc.tool_name,
                                content="(cancelled)",
                                tool_call_id=tc.tool_call_id,
                            )
                            for tc in missing
                        ],
                    )
                    repaired.append(synthetic)
                    new_messages.append(synthetic)
                    i += 1
                    continue
        if repaired is not None:
            repaired.append(msg)
        i += 1

    if repaired is not None:
        return repaired, all_tool_names, new_messages
    return history, [], []


# ── Compaction helpers ───────────────────────────────────────────────

_SUMMARY_MARKER = "[Context summary — earlier conversation was compacted]"


def _is_user_turn_start(msg: ModelMessage) -> bool:
    """True if *msg* begins a new user turn.

    A turn starts with a ``ModelRequest`` containing at least one
    ``UserPromptPart`` (as opposed to tool-return-only requests).
    """
    if not isinstance(msg, ModelRequest):
        return False
    return any(isinstance(p, UserPromptPart) for p in msg.parts)


def _collect_tool_call_ids(messages: list[ModelMessage]) -> set[str]:
    """Return the set of ``tool_call_id`` values from all ``ToolCallPart``\\s.

    Used by ``split_history`` to determine which tool returns in the
    *kept* partition still have their matching call.
    """
    ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id:
                    ids.add(part.tool_call_id)
    return ids


def _is_orphaned_tool_return(msg: ModelMessage, known_call_ids: set[str]) -> bool:
    """True if *msg* is a tool-return-only request with no matching call.

    A message is orphaned when:
    1. It is a ``ModelRequest`` with no ``UserPromptPart``.
    2. It contains only ``ToolReturnPart``\\s.
    3. None of those return IDs appear in *known_call_ids*.

    Messages with a ``UserPromptPart`` are never orphaned — they
    carry user intent regardless of any tool returns alongside them.
    """
    if not isinstance(msg, ModelRequest):
        return False
    if any(isinstance(p, UserPromptPart) for p in msg.parts):
        return False
    tool_returns = [
        p for p in msg.parts if isinstance(p, ToolReturnPart) and p.tool_call_id is not None
    ]
    if not tool_returns:
        return False
    return all(p.tool_call_id not in known_call_ids for p in tool_returns)


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


def snap_to_safe_boundary(messages: list[ModelMessage], count: int) -> int:
    """Adjust *count* so a split at that index never separates a
    ``ModelResponse`` (tool calls) from its ``ModelRequest`` (tool results).

    A ``ModelResponse`` with ``ToolCallPart``\\s must stay in the same
    partition as the immediately following ``ModelRequest``.  If
    *count* would split them, it is reduced until the boundary is safe.

    Returns the adjusted count (may be 0 if no safe split exists).
    """
    while count > 0:
        prev = messages[count - 1]
        if isinstance(prev, ModelResponse) and any(isinstance(p, ToolCallPart) for p in prev.parts):
            count -= 1
        else:
            break
    return count


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
