"""Message and model builders for tests.

Message builders produce structurally valid messages matching what
the LLM pipeline actually produces.  Model builders produce
configured ``TestModel`` / ``FunctionModel`` instances for common
patterns.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from rbtr.engine.core import Engine

_USAGE = RequestUsage(input_tokens=0, output_tokens=0)


def _user(text: str) -> ModelRequest:
    """User prompt → ``ModelRequest[UserPromptPart]``."""
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    """Text-only response → ``ModelResponse[TextPart]``."""
    return ModelResponse(parts=[TextPart(content=text)], usage=_USAGE, model_name="test")


def _resp(*parts: TextPart | ToolCallPart | ThinkingPart) -> ModelResponse:
    """Multi-part response → ``ModelResponse[...]``."""
    return ModelResponse(parts=list(parts), usage=_USAGE, model_name="test")


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


# ── Model builders ──────────────────────────────────────────────────


def _text_model(text: str) -> TestModel:
    """TestModel that returns *text* without calling tools."""
    return TestModel(custom_output_text=text, call_tools=[])


def _tool_model() -> TestModel:
    """TestModel that calls all registered tools."""
    return TestModel(call_tools="all")


def _streaming_model(*chunks: str) -> FunctionModel:
    """FunctionModel that streams *chunks* as text."""

    async def _stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    return FunctionModel(stream_function=_stream)


def _tool_then_text_model() -> FunctionModel:
    """FunctionModel: tool call on first request, text on second.

    Stateful per instance.  Provides both ``function`` and
    ``stream_function`` so the agent's streaming iteration works.
    """
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        usage = RequestUsage(input_tokens=50, output_tokens=10)
        if call_count == 1 and info.function_tools:
            tool = info.function_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool_name=tool.name, args="{}")],
                usage=usage,
                model_name="test-fn",
            )
        return ModelResponse(
            parts=[TextPart(content="done")],
            usage=usage,
            model_name="test-fn",
        )

    async def stream_fn(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1 and info.function_tools:
            tool = info.function_tools[0]
            yield {0: DeltaToolCall(name=tool.name, json_args="{}")}
        else:
            yield "done"

    return FunctionModel(model_fn, stream_function=stream_fn)
