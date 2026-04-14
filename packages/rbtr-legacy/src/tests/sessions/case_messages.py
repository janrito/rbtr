"""Single-message cases for store round-trip tests.

Each ``@case`` returns one ``ModelRequest`` or ``ModelResponse``.
Used by ``test_store.py`` to verify that every message/part type
survives save → load losslessly.
"""

from __future__ import annotations

import base64

from pydantic_ai.messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage
from pytest_cases import case

_USAGE = RequestUsage(input_tokens=100, output_tokens=50)
_USAGE_LARGE = RequestUsage(input_tokens=5000, output_tokens=200)

# A tiny 1x1 red PNG for FilePart tests.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "2mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)


# ── Requests ─────────────────────────────────────────────────────────


@case(tags=["request"])
def case_user_prompt() -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content="explain closures")])


@case(tags=["request"])
def case_system_prompt() -> ModelRequest:
    return ModelRequest(parts=[SystemPromptPart(content="be helpful")])


@case(tags=["request"])
def case_tool_return() -> ModelRequest:
    from pydantic_ai.messages import ToolReturnPart

    return ModelRequest(
        parts=[
            ToolReturnPart(tool_name="read_file", content="def hello(): ...", tool_call_id="tc1")
        ]
    )


@case(tags=["request"])
def case_retry_prompt() -> ModelRequest:
    return ModelRequest(
        parts=[RetryPromptPart(content="try again", tool_name="f", tool_call_id="tc1")]
    )


# ── Responses — basic ────────────────────────────────────────────────


@case(tags=["response"])
def case_text_response() -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content="A closure captures variables.")],
        usage=_USAGE,
        model_name="test",
    )


@case(tags=["response"])
def case_tool_call() -> ModelResponse:
    return ModelResponse(
        parts=[
            TextPart(content="Let me check."),
            ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
        ],
        usage=_USAGE_LARGE,
        model_name="test",
    )


@case(tags=["response"])
def case_thinking() -> ModelResponse:
    return ModelResponse(
        parts=[ThinkingPart(content="hmm..."), TextPart(content="Here's my answer.")],
        usage=_USAGE_LARGE,
        model_name="test",
    )


@case(tags=["response"])
def case_multi_tool() -> ModelResponse:
    return ModelResponse(
        parts=[
            TextPart(content="Let me check."),
            ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="tc1"),
            ToolCallPart(tool_name="grep", args={"pattern": "TODO"}, tool_call_id="tc2"),
        ],
        usage=_USAGE_LARGE,
        model_name="test",
    )


@case(tags=["response"])
def case_zero_usage() -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content="hi")],
        model_name="test",
    )


# ── Responses — extended part types ──────────────────────────────────


@case(tags=["response", "file"])
def case_file_image() -> ModelResponse:
    return ModelResponse(
        parts=[
            FilePart(content=BinaryContent(data=_TINY_PNG, media_type="image/png")),
            TextPart(content="A 1x1 red pixel."),
        ],
        usage=_USAGE,
        model_name="test",
    )


@case(tags=["response"])
def case_builtin_tool_call() -> ModelResponse:
    return ModelResponse(
        parts=[
            BuiltinToolCallPart(
                tool_name="web_search",
                args={"query": "weather today"},
                tool_call_id="bt1",
            ),
        ],
        usage=_USAGE,
        model_name="test",
    )


@case(tags=["response"])
def case_builtin_tool_return() -> ModelResponse:
    return ModelResponse(
        parts=[
            BuiltinToolReturnPart(
                tool_name="web_search",
                content="Sunny, 72°F",
                tool_call_id="bt1",
            ),
        ],
        usage=_USAGE,
        model_name="test",
    )


@case(tags=["response", "file"])
def case_mixed_file_and_text() -> ModelResponse:
    return ModelResponse(
        parts=[
            ThinkingPart(content="analysing the image"),
            FilePart(content=BinaryContent(data=_TINY_PNG, media_type="image/png")),
            TextPart(content="review complete"),
        ],
        usage=_USAGE,
        model_name="test",
    )


@case(tags=["response"])
def case_mixed_thinking_tool_text() -> ModelResponse:
    return ModelResponse(
        parts=[
            ThinkingPart(content="need to search first"),
            ToolCallPart(tool_name="grep", args={"pattern": "TODO"}, tool_call_id="tc1"),
            TextPart(content="let me check"),
        ],
        usage=_USAGE_LARGE,
        model_name="test",
    )
