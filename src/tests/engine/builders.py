"""Message builders for engine tests.

Each builder produces one structurally valid message matching
what the LLM pipeline actually produces.  For complex messages
(parallel tools, combined returns), spell out
``ModelResponse(parts=[...])`` / ``ModelRequest(parts=[...])``
inline at the call site.
"""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from rbtr.engine.core import Engine

_USAGE = RequestUsage(input_tokens=0, output_tokens=0)


def _user(text: str) -> ModelRequest:
    """User prompt → ``ModelRequest[UserPromptPart]``."""
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    """Text-only response → ``ModelResponse[TextPart]``."""
    return ModelResponse(parts=[TextPart(content=text)], usage=_USAGE, model_name="test")


def _tool_turn(
    name: str,
    args: dict[str, str] | None = None,
    *,
    preamble: str = "",
    call_id: str = "call_0",
) -> ModelResponse:
    """Text + tool call in one response — matches real LLM output.

    ``ModelResponse[TextPart, ToolCallPart]``
    """
    parts: list[TextPart | ToolCallPart] = []
    if preamble:
        parts.append(TextPart(content=preamble))
    parts.append(ToolCallPart(tool_name=name, args=args or {}, tool_call_id=call_id))
    return ModelResponse(parts=parts, usage=_USAGE, model_name="test")


def _tool_result(name: str, content: str, *, call_id: str = "call_0") -> ModelRequest:
    """Tool return → ``ModelRequest[ToolReturnPart]``."""
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=name, content=content, tool_call_id=call_id)]
    )


def _tool_retry(name: str, error: str, *, call_id: str = "call_0") -> ModelRequest:
    """Failed tool → ``ModelRequest[RetryPromptPart]``."""
    return ModelRequest(
        parts=[RetryPromptPart(content=error, tool_name=name, tool_call_id=call_id)]
    )


def _tool_call_only(
    name: str, args: dict[str, str] | None = None, *, call_id: str = "call_0"
) -> ModelResponse:
    """Standalone tool-call response with NO text preamble.

    **For algorithm edge-case tests only** (split, orphan, snap).
    Normal test data should use ``_tool_turn`` which includes a text
    preamble matching real LLM output.
    """
    return ModelResponse(
        parts=[ToolCallPart(tool_name=name, args=args or {}, tool_call_id=call_id)],
        usage=_USAGE,
        model_name="test",
    )


def _thinking(text: str) -> ModelResponse:
    """Thinking-only response → ``ModelResponse[ThinkingPart]``."""
    return ModelResponse(parts=[ThinkingPart(content=text)], usage=_USAGE, model_name="test")


def _turns(n: int) -> list[ModelRequest | ModelResponse]:
    """Create *n* user→assistant turn pairs."""
    msgs: list[ModelRequest | ModelResponse] = []
    for i in range(n):
        msgs.append(_user(f"question {i}"))
        msgs.append(_assistant(f"answer {i}"))
    return msgs


def _seed(
    engine: Engine,
    messages: list[ModelRequest | ModelResponse],
    *,
    repo_owner: str | None = None,
    repo_name: str | None = None,
    model_name: str | None = None,
    cost: float | None = None,
) -> None:
    """Seed messages into the engine's store."""
    engine._sync_store_context()
    engine.store.save_messages(
        engine.state.session_id,
        messages,
        repo_owner=repo_owner,
        repo_name=repo_name,
        model_name=model_name,
        cost=cost,
    )
