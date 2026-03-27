"""Live engine scenarios for replay roundtrip tests.

Each ``@case`` returns a ``Scenario`` — a sequence of (model, prompt)
turns with optional compaction.  The test function executes them.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.models import Model
from pytest_cases import case

from tests.engine.builders import _streaming_model, _text_model, _tool_model


@dataclass(frozen=True, slots=True)
class Scenario:
    """A sequence of (model, prompt) turns with optional compaction."""

    turns: list[tuple[Model, str]]
    compact_after: int | None = None
    expect_tools: bool = False
    expect_compaction_marker: bool = False


@case(tags=["live"])
def case_single_text_turn() -> Scenario:
    """Minimal: one prompt, one response."""
    return Scenario(turns=[(_text_model("Hello!"), "hi")])


@case(tags=["live"])
def case_multi_turn_text() -> Scenario:
    """Two text-only turns."""
    return Scenario(
        turns=[
            (_text_model("First."), "question 1"),
            (_text_model("Second."), "question 2"),
        ],
    )


@case(tags=["live"])
def case_streaming_text() -> Scenario:
    """Streaming text via FunctionModel."""
    return Scenario(
        turns=[(_streaming_model("Hello ", "from ", "streaming!"), "greet me")],
    )


@case(tags=["live", "tool"])
def case_tool_calls() -> Scenario:
    """Tool-calling turn."""
    return Scenario(
        turns=[(_tool_model(), "check the code")],
        expect_tools=True,
    )


@case(tags=["live", "tool"])
def case_tool_then_text() -> Scenario:
    """Tool call followed by a text-only follow-up."""
    return Scenario(
        turns=[
            (_tool_model(), "read the file"),
            (_text_model("Got it."), "explain it"),
        ],
        expect_tools=True,
    )


@case(tags=["live", "compaction"])
def case_compaction() -> Scenario:
    """Multiple turns then compaction."""
    return Scenario(
        turns=[(_text_model(f"Answer {i}."), f"question {i}") for i in range(5)],
        compact_after=5,
        expect_compaction_marker=True,
    )


@case(tags=["live", "compaction"])
def case_compaction_then_continue() -> Scenario:
    """Compact, then continue with more turns."""
    turns: list[tuple[Model, str]] = [
        (_text_model(f"Answer {i}."), f"question {i}") for i in range(5)
    ]
    turns.append((_text_model("After compaction."), "follow up"))
    return Scenario(
        turns=turns,
        compact_after=5,
        expect_compaction_marker=True,
    )


@case(tags=["live", "tool", "compaction"])
def case_tools_then_compaction() -> Scenario:
    """Tool calls followed by text turns then compaction."""
    turns: list[tuple[Model, str]] = [(_tool_model(), "check code")]
    for i in range(4):
        turns.append((_text_model(f"Note {i}."), f"q{i}"))
    return Scenario(
        turns=turns,
        compact_after=5,
        expect_tools=True,
        expect_compaction_marker=True,
    )
