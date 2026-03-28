"""Shared conversation history cases for `pytest-cases`.

Each ``@case`` function returns a ``list[ModelMessage]`` representing
a realistic conversation.  All cases are structurally valid: messages
alternate ``ModelRequest`` / ``ModelResponse``, tool calls have
matching returns, combined parts in single messages.

Consumed by tests across ``tests/sessions/``, ``tests/engine/``, and
``tests/llm/`` via ``@parametrize_with_cases``.
"""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pytest_cases import case

from tests.engine.builders import _assistant, _resp, _tool_result, _tool_turn, _user

# ── Text-only ────────────────────────────────────────────────────────


@case(tags=["conversation", "compactable"])
def case_text_only() -> list[ModelMessage]:
    """Text-only multi-turn."""
    return [
        _user("Review PR #42"),
        _assistant("I'll start by looking at the diff."),
        _user("What about test coverage?"),
        _assistant("Tests look adequate."),
        _user("Any security concerns?"),
        _assistant("The import os is unused."),
        _user("Thanks, ship it."),
        _assistant("LGTM, approved."),
    ]


@case(tags=["conversation"])
def case_single_turn() -> list[ModelMessage]:
    """Single turn — minimum viable conversation."""
    return [
        _user("hello"),
        _assistant("Hi there!"),
    ]


# ── Tool calls ───────────────────────────────────────────────────────


@case(tags=["conversation", "tool", "compactable"])
def case_single_tool() -> list[ModelMessage]:
    """Single tool call per turn — the common case."""
    return [
        _user("Read foo.py"),
        _resp(
            TextPart(content="Let me read that."),
            ToolCallPart(tool_name="read_file", args={"path": "foo.py"}, tool_call_id="c1"),
        ),
        _tool_result("read_file", "def main(): pass", call_id="c1"),
        _assistant("foo.py has one function."),
        _user("Now check bar.py"),
        _resp(
            TextPart(content="Reading bar.py."),
            ToolCallPart(tool_name="read_file", args={"path": "bar.py"}, tool_call_id="c2"),
        ),
        _tool_result("read_file", "import foo", call_id="c2"),
        _assistant("bar.py imports foo."),
        _user("Any issues?"),
        _assistant("No issues found."),
    ]


@case(tags=["conversation", "tool", "compactable"])
def case_parallel_tools() -> list[ModelMessage]:
    """Parallel tool calls (2 calls in one response)."""
    return [
        _user("Check both files"),
        _resp(
            TextPart(content="Reading both."),
            ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="c1"),
            ToolCallPart(tool_name="read_file", args={"path": "b.py"}, tool_call_id="c2"),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="def a(): ...", tool_call_id="c1"),
                ToolReturnPart(tool_name="read_file", content="def b(): ...", tool_call_id="c2"),
            ]
        ),
        _assistant("Both files read successfully."),
        _user("Any differences?"),
        _assistant("They have different functions."),
        _user("Thanks"),
        _assistant("You're welcome."),
    ]


@case(tags=["conversation", "tool", "compactable"])
def case_chained_tools() -> list[ModelMessage]:
    """Chained tool calls (call → return → call → return in one turn)."""
    return [
        _user("Find TODOs and read the files"),
        _resp(
            TextPart(content="Searching for TODOs."),
            ToolCallPart(tool_name="grep", args={"pattern": "TODO"}, tool_call_id="c1"),
        ),
        _tool_result("grep", "foo.py:10 TODO fix this", call_id="c1"),
        _resp(
            TextPart(content="Found one. Reading the file."),
            ToolCallPart(tool_name="read_file", args={"path": "foo.py"}, tool_call_id="c2"),
        ),
        _tool_result("read_file", "def main(): pass  # TODO fix this", call_id="c2"),
        _assistant("The TODO is in main()."),
        _user("Fix it"),
        _assistant("Done."),
        _user("Verify"),
        _assistant("All clean."),
    ]


@case(tags=["conversation", "tool", "failure", "compactable"])
def case_tool_failure() -> list[ModelMessage]:
    """Failed tool (RetryPromptPart)."""
    return [
        _user("Read secret.py"),
        _resp(
            TextPart(content="Reading."),
            ToolCallPart(tool_name="read_file", args={"path": "secret.py"}, tool_call_id="c1"),
        ),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    content="Permission denied: secret.py",
                    tool_name="read_file",
                    tool_call_id="c1",
                ),
            ]
        ),
        _assistant("I can't read secret.py — permission denied."),
        _user("Try config.py instead"),
        _resp(
            ToolCallPart(tool_name="read_file", args={"path": "config.py"}, tool_call_id="c2"),
        ),
        _tool_result("read_file", "DEBUG=True", call_id="c2"),
        _assistant("config.py sets DEBUG=True."),
    ]


@case(tags=["conversation", "tool", "thinking", "compactable"])
def case_thinking_with_tools() -> list[ModelMessage]:
    """Thinking + tool call + text in one response."""
    return [
        _user("Analyse the codebase"),
        _resp(
            ThinkingPart(content="Let me think about the structure..."),
            TextPart(content="I'll start with the main module."),
            ToolCallPart(tool_name="read_file", args={"path": "main.py"}, tool_call_id="c1"),
        ),
        _tool_result("read_file", "from app import run\nrun()", call_id="c1"),
        _resp(
            ThinkingPart(content="Simple entry point..."),
            TextPart(content="Now checking app.py."),
            ToolCallPart(tool_name="read_file", args={"path": "app.py"}, tool_call_id="c2"),
        ),
        _tool_result("read_file", "def run(): print('hello')", call_id="c2"),
        _assistant("The app is a simple hello-world."),
        _user("Any improvements?"),
        _assistant("Consider adding error handling."),
    ]


@case(tags=["conversation", "tool", "compactable"])
def case_reordered_returns() -> list[ModelMessage]:
    """Returns in different order than calls."""
    return [
        _user("Read both files"),
        _resp(
            TextPart(content="Reading both."),
            ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="c1"),
            ToolCallPart(tool_name="read_file", args={"path": "b.py"}, tool_call_id="c2"),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="def b(): ...", tool_call_id="c2"),
                ToolReturnPart(tool_name="read_file", content="def a(): ...", tool_call_id="c1"),
            ]
        ),
        _assistant("Both read. b.py returned first."),
        _user("Why?"),
        _assistant("Async execution order."),
        _user("Ok"),
        _assistant("Moving on."),
    ]


@case(tags=["conversation", "tool", "compactable"])
def case_tool_no_preamble() -> list[ModelMessage]:
    """Tool call with no text preamble (model goes straight to tool)."""
    return [
        _user("Read a.py"),
        _resp(
            ToolCallPart(tool_name="read_file", args={"path": "a.py"}, tool_call_id="c1"),
        ),
        _tool_result("read_file", "content", call_id="c1"),
        _assistant("Here it is."),
        _user("Thanks"),
        _assistant("Done."),
    ]


@case(tags=["conversation", "compaction", "replay_only"])
def case_post_compaction() -> list[ModelMessage]:
    """Post-compaction — summary message + kept turns."""
    return [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content="[Context summary — earlier conversation was compacted]\n\n"
                    "Discussed PR #42. Found unused import."
                )
            ]
        ),
        _assistant("Continuing from the summary."),
        _user("Any other issues?"),
        _assistant("No, looks clean."),
        _user("Ship it"),
        _assistant("LGTM, approved."),
    ]


@case(tags=["conversation", "tool", "failure", "compactable"])
def case_mixed_parallel_outcome() -> list[ModelMessage]:
    """Parallel tools — one succeeds, one fails."""
    return [
        _user("Read both files"),
        _resp(
            TextPart(content="Reading both."),
            ToolCallPart(tool_name="read_file", args={"path": "ok.py"}, tool_call_id="c1"),
            ToolCallPart(tool_name="read_file", args={"path": "secret.py"}, tool_call_id="c2"),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="read_file", content="def ok(): ...", tool_call_id="c1"),
                RetryPromptPart(
                    content="Permission denied: secret.py",
                    tool_name="read_file",
                    tool_call_id="c2",
                ),
            ]
        ),
        _assistant("ok.py read successfully. secret.py is not accessible."),
        _user("Just use ok.py then"),
        _assistant("Will do."),
    ]


@case(tags=["conversation", "tool", "compactable"])
def case_empty_tool_args() -> list[ModelMessage]:
    """Tool call with empty args."""
    return [
        _user("List all files"),
        _tool_turn("list_files", {}, preamble="Listing.", call_id="c1"),
        _tool_result("list_files", "a.py\nb.py", call_id="c1"),
        _assistant("Found 2 files."),
        _user("Thanks"),
        _assistant("Done."),
    ]


# ── Thinking + tools ─────────────────────────────────────────────────


@case(tags=["conversation", "tool", "thinking", "compactable"])
def case_thinking_tool_multi_turn() -> list[ModelMessage]:
    """Thinking + tool call + tool return + text follow-up.

    Multi-turn: tool-calling turn with text preamble, then a
    thinking response, then a text-only Q/A turn.
    """
    return [
        _user("review src/tui.py"),
        _resp(
            TextPart(content="I'll read the file."),
            ToolCallPart(tool_name="read_file", args={"path": "src/tui.py"}, tool_call_id="c1"),
        ),
        _tool_result("read_file", "class TUI:\n    pass", call_id="c1"),
        _resp(
            ThinkingPart(content="The TUI class is minimal..."),
            TextPart(content="Here's my analysis."),
        ),
        _user("any security issues?"),
        _assistant("No security issues found."),
    ]


@case(tags=["conversation", "tool", "thinking", "compactable"])
def case_signed_thinking_multi_turn() -> list[ModelMessage]:
    """Signed thinking + parallel tool calls + chained tool rounds.

    Exercises provider-metadata fidelity: ``ThinkingPart`` with
    ``signature`` and ``provider_name`` fields that Anthropic
    requires on round-trip.
    """
    return [
        # Turn 1: thinking + tool call → return → thinking + parallel calls → returns → thinking + text
        _user("review the draft"),
        _resp(
            ThinkingPart(
                content="User wants a review. Let me find the draft.",
                signature="sig_t1",
                provider_name="anthropic",
            ),
            ToolCallPart(tool_name="list_files", args={"pattern": "draft"}, tool_call_id="c1"),
        ),
        _tool_result("list_files", "draft.md\nnotes.md", call_id="c1"),
        _resp(
            ThinkingPart(
                content="Found the draft. Let me read it.",
                signature="sig_t2",
                provider_name="anthropic",
            ),
            ToolCallPart(tool_name="read_file", args={"path": "draft.md"}, tool_call_id="c2"),
            ToolCallPart(tool_name="read_file", args={"path": "notes.md"}, tool_call_id="c3"),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="read_file",
                    content="# Draft\n\nContent here.",
                    tool_call_id="c2",
                ),
                ToolReturnPart(
                    tool_name="read_file",
                    content="# Notes\n\nReview notes.",
                    tool_call_id="c3",
                ),
            ]
        ),
        _resp(
            ThinkingPart(
                content="Analysing both documents for cohesion.",
                signature="sig_t3",
                provider_name="anthropic",
            ),
            TextPart(content="Here's my analysis of the draft."),
        ),
        # Turn 2: follow-up → thinking + text
        _user("any issues with section 3?"),
        _resp(
            ThinkingPart(
                content="Let me check section 3 specifically.",
                signature="sig_t4",
                provider_name="anthropic",
            ),
            TextPart(content="Section 3 has a coherence issue."),
        ),
    ]


@case(tags=["conversation", "tool", "compactable"])
def case_tool_boundary_straddle() -> list[ModelMessage]:
    """Tool return straddling a turn boundary.

    Turn 3 ends with a tool call; its return arrives in the same
    ``ModelRequest`` as turn 4's user prompt.  Compaction at the
    turn boundary would orphan the return without special handling.
    Reproduces a production bug.
    """
    return [
        # Turn 1: single tool call
        _user("find all TODO comments"),
        _resp(
            TextPart(content="Searching..."),
            ToolCallPart(tool_name="grep", args={"pattern": "TODO"}, tool_call_id="c1"),
        ),
        _tool_result("grep", "foo.py:10 TODO\nbar.py:3 TODO", call_id="c1"),
        _assistant("Found 2 TODOs."),
        # Turn 2: chained tool calls
        _user("show me foo.py"),
        _resp(
            TextPart(content="Reading foo.py."),
            ToolCallPart(tool_name="read_file", args={"path": "foo.py"}, tool_call_id="c2"),
        ),
        _tool_result("read_file", "def main(): pass", call_id="c2"),
        _resp(
            TextPart(content="Checking references."),
            ToolCallPart(tool_name="grep", args={"pattern": "main"}, tool_call_id="c3"),
        ),
        _tool_result("grep", "foo.py:1 def main()", call_id="c3"),
        _assistant("foo.py has one function."),
        # Turn 3: text + tool call (return arrives in turn 4's request)
        _user("what other files?"),
        _resp(
            TextPart(content="Let me check."),
            ToolCallPart(tool_name="list_files", args={"path": "."}, tool_call_id="c4"),
        ),
        # Turn 4: straddling — tool return + user prompt in one request
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="list_files",
                    content="foo.py\nbar.py",
                    tool_call_id="c4",
                ),
                UserPromptPart(content="any security concerns?"),
            ]
        ),
        _assistant("No issues."),
        # Turn 5
        _user("summarise"),
        _assistant("All good."),
    ]
