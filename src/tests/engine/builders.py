"""Message builders for engine tests."""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from rbtr.engine.core import Engine
from rbtr.llm.compact import _SummaryResult

_USAGE = RequestUsage(input_tokens=0, output_tokens=0)


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)], usage=_USAGE, model_name="test")


def _tool_return(name: str, content: str, *, call_id: str = "call_0") -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=name, content=content, tool_call_id=call_id)]
    )


def _tool_call(
    name: str, args: dict[str, str] | None = None, *, call_id: str = "call_0"
) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=name, args=args or {}, tool_call_id=call_id)],
        usage=_USAGE,
        model_name="test",
    )


def _thinking(text: str) -> ModelResponse:
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


def summary_result(text: str = "Summary.") -> _SummaryResult:
    """Build a `_SummaryResult` with zero cost — for mocking `_stream_summary`."""
    return _SummaryResult(text=text, input_tokens=0, output_tokens=0, cost=0.0)
