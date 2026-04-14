"""Tests for context markers — expansion, shell, submit, and dismiss.

Context markers live on `InputState.context_regions`, outside the
editing buffer.  They are rendered above the prompt and dismissed
via Backspace at cursor position 0.
"""

from __future__ import annotations

import queue
from collections.abc import Generator

import pytest
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys
from pydantic_ai.messages import ModelRequest, UserPromptPart

from rbtr_legacy.config import config
from rbtr_legacy.engine.core import Engine
from rbtr_legacy.engine.shell import _emit_shell_context
from rbtr_legacy.engine.types import TaskType
from rbtr_legacy.events import ContextMarkerReady, Event
from rbtr_legacy.sessions.store import SessionStore
from rbtr_legacy.shell_exec import ShellResult
from rbtr_legacy.state import EngineState
from rbtr_legacy.tui.input import InputReader, InputState, PasteRegion
from tests.helpers import drain

# ── Helpers ──────────────────────────────────────────────────────────


def _context_events(events: queue.Queue[Event]) -> list[ContextMarkerReady]:
    """Drain the queue and return only `ContextMarkerReady` events."""
    return [e for e in drain(events) if isinstance(e, ContextMarkerReady)]


@pytest.fixture
def shell_engine() -> Generator[Engine]:
    """Engine wired for shell context tests."""
    state = EngineState(owner="testowner", repo_name="testrepo")
    events: queue.Queue[Event] = queue.Queue()
    with Engine(state, events, store=SessionStore()) as eng:
        yield eng


def test_context_not_in_marker_spans(input_state: InputState) -> None:
    """Context regions are outside the buffer — marker_spans doesn't see them."""
    input_state.add_context("[/model gpt-4o]", "Switched.")
    assert input_state.marker_spans() == []


def test_context_pop_removes_last(input_state: InputState) -> None:
    input_state.add_context("[/review]", "Selected.")
    input_state.add_context("[/model]", "Switched.")
    popped = input_state.pop_context()
    assert popped is not None
    assert popped.marker == "[/model]"
    assert len(input_state.context_regions) == 1


# ── Adding context does not mutate the buffer ────────────────────────


def test_add_context_empty_buffer_unchanged(input_state: InputState) -> None:
    input_state.add_context("[/review → PR #1]", "Selected PR.")
    assert input_state.text == ""
    assert len(input_state.context_regions) == 1


def test_add_context_existing_text_unchanged(input_state: InputState) -> None:
    input_state.set_text("hello world", cursor=5)
    input_state.add_context("[/model → gpt-4o]", "Switched.")
    assert input_state.text == "hello world"
    assert input_state.cursor == 5


def test_add_context_multiple_in_order(input_state: InputState) -> None:
    input_state.add_context("[/review → PR #1]", "PR selected.")
    input_state.add_context("[/model → gpt-4o]", "Switched.")
    assert len(input_state.context_regions) == 2
    assert input_state.context_regions[0].marker == "[/review → PR #1]"
    assert input_state.context_regions[1].marker == "[/model → gpt-4o]"
    assert input_state.text == ""


def test_add_context_cursor_preserved(input_state: InputState) -> None:
    """User is typing at position 5; adding context doesn't move cursor."""
    input_state.set_text("hello world", cursor=5)
    input_state.add_context("[/review → PR #1]", "Selected.")
    assert input_state.cursor == 5
    assert input_state.text == "hello world"


# ── Expansion format on submit ───────────────────────────────────────


def test_expand_no_markers_no_prefix(input_state: InputState) -> None:
    input_state.set_text("just a question")
    assert input_state.expand_markers(input_state.text) == "just a question"


def test_expand_single_context_marker(input_state: InputState) -> None:
    input_state.set_text("what changed?")
    input_state.add_context("[/review → PR #42]", "Selected PR #42.")
    result = input_state.expand_markers(input_state.text)
    assert result == (
        "<recent_actions>\n"
        "The user ran the following commands since the last message:\n"
        "- Selected PR #42.\n"
        "</recent_actions>\n"
        "\n"
        "what changed?"
    )


def test_expand_multiple_context_markers(input_state: InputState) -> None:
    input_state.set_text("summarise")
    input_state.add_context("[/review → PR #42]", "Selected PR #42.")
    input_state.add_context("[/model → gpt-4o]", "Switched model.")
    result = input_state.expand_markers(input_state.text)
    assert "<recent_actions>" in result
    assert "- Selected PR #42." in result
    assert "- Switched model." in result
    assert result.endswith("</recent_actions>\n\nsummarise")


def test_expand_context_markers_only_no_user_text(input_state: InputState) -> None:
    input_state.add_context("[/review → PR #42]", "Selected PR #42.")
    result = input_state.expand_markers(input_state.text)
    assert result == (
        "<recent_actions>\n"
        "The user ran the following commands since the last message:\n"
        "- Selected PR #42.\n"
        "</recent_actions>"
    )


def test_expand_mixed_paste_and_context(input_state: InputState) -> None:
    input_state.set_text("see [pasted 4 lines] above")
    input_state.add_context("[/review → PR #42]", "Selected PR #42.")
    input_state.paste_regions.append(PasteRegion(marker="[pasted 4 lines]", content="a\nb\nc\nd"))
    result = input_state.expand_markers(input_state.text)
    assert result.startswith("<recent_actions>\n")
    assert "- Selected PR #42." in result
    assert "</recent_actions>\n\n" in result
    assert "a\nb\nc\nd" in result
    assert "[pasted 4 lines]" not in result


def test_expand_user_deletes_all_context_markers(input_state: InputState) -> None:
    """If user dismisses all context markers, message has no prefix."""
    input_state.set_text("just a question")
    assert input_state.expand_markers(input_state.text) == "just a question"


def test_expand_context_order_preserved(input_state: InputState) -> None:
    """Context entries appear in insertion order (chronological)."""
    input_state.set_text("question")
    input_state.add_context("[B]", "Second command.")
    input_state.add_context("[A]", "First command.")
    result = input_state.expand_markers(input_state.text)
    lines = result.split("\n")
    action_lines = [line for line in lines if line.startswith("- ")]
    assert action_lines[0] == "- Second command."
    assert action_lines[1] == "- First command."


# ── Shell context markers ────────────────────────────────────────────


def test_shell_success_emits_context(shell_engine: Engine) -> None:
    shell_engine.run_task(TaskType.SHELL, "echo hello")
    markers = _context_events(shell_engine.events)
    assert len(markers) == 1
    assert "echo hello" in markers[0].marker
    assert "exit 0" in markers[0].marker
    assert "hello" in markers[0].content


def test_shell_failure_includes_exit_code(shell_engine: Engine) -> None:
    shell_engine.run_task(TaskType.SHELL, "false")
    markers = _context_events(shell_engine.events)
    assert len(markers) == 1
    assert "exit 1" in markers[0].marker
    assert "exit code 1" in markers[0].content


def test_shell_stderr_included(shell_engine: Engine) -> None:
    shell_engine.run_task(TaskType.SHELL, "echo err >&2")
    markers = _context_events(shell_engine.events)
    assert len(markers) == 1
    assert "(stderr)" in markers[0].content
    assert "err" in markers[0].content


def test_shell_empty_command_no_context(shell_engine: Engine) -> None:
    """Empty shell command shows usage, no context marker."""
    shell_engine.run_task(TaskType.SHELL, "")
    markers = _context_events(shell_engine.events)
    assert markers == []


# ── Shell context truncation ─────────────────────────────────────────


def test_shell_truncation_applied(shell_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.tui, "shell_context_max_chars", 50)
    shell_engine.run_task(TaskType.SHELL, "seq 1 1000")
    markers = _context_events(shell_engine.events)
    assert len(markers) == 1
    assert "… (truncated)" in markers[0].content
    body_before_suffix = markers[0].content.split("\n… (truncated)")[0]
    assert len(body_before_suffix) <= 50


def test_shell_no_truncation_when_under_limit(shell_engine: Engine) -> None:
    shell_engine.run_task(TaskType.SHELL, "echo hi")
    markers = _context_events(shell_engine.events)
    assert len(markers) == 1
    assert "… (truncated)" not in markers[0].content


# ── _emit_shell_context unit tests ───────────────────────────────────


def test_emit_shell_context_success_format(shell_engine: Engine) -> None:
    _emit_shell_context(shell_engine, "git status", ShellResult("clean", "", 0))
    markers = _context_events(shell_engine.events)
    assert len(markers) == 1
    assert markers[0].marker == "[! git status — exit 0]"
    assert "$ git status" in markers[0].content
    assert "clean" in markers[0].content
    assert "exit code 0" in markers[0].content


def test_emit_shell_context_stderr_format(shell_engine: Engine) -> None:
    _emit_shell_context(shell_engine, "bad", ShellResult("", "not found", 127))
    markers = _context_events(shell_engine.events)
    assert markers[0].marker == "[! bad — exit 127]"
    assert "(stderr)\nnot found" in markers[0].content


def test_emit_shell_context_empty_output(shell_engine: Engine) -> None:
    _emit_shell_context(shell_engine, "true", ShellResult("", "", 0))
    markers = _context_events(shell_engine.events)
    assert len(markers) == 1
    assert "$ true" in markers[0].content


def test_emit_shell_context_truncation(
    shell_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config.tui, "shell_context_max_chars", 30)
    _emit_shell_context(shell_engine, "cmd", ShellResult("x" * 100, "", 0))
    markers = _context_events(shell_engine.events)
    assert "… (truncated)" in markers[0].content


# ── Round-trip: expanded context survives save/load ──────────────────


def test_expanded_message_round_trips(input_state: InputState) -> None:
    """Expanded context in user messages round-trips through the store."""

    input_state.set_text("what changed?")
    input_state.add_context("[/review → PR #42]", "Selected PR #42.")
    text = input_state.expand_markers(input_state.text)

    with SessionStore() as store:
        sid = "test-session"
        store.set_context(sid)
        msg = ModelRequest(parts=[UserPromptPart(content=text)])
        store.save_messages(sid, [msg])
        loaded = store.load_messages(sid)

        assert len(loaded) == 1
        assert loaded[0].parts[0].content == text  # type: ignore[union-attr]  # UserPromptPart.content


# ── Submit: slash/shell commands bypass marker expansion ─────────────


def test_submit_slash_command_preserves_context(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Enter with context + `/help` submits `/help`; context survives."""
    input_state.set_text("/help")
    input_state.add_context("[/review]", "Selected PR.")

    input_reader._on_key(KeyPress(Keys.Enter, "\r"))

    submitted = input_state.submitted.get_nowait()
    assert submitted == "/help"
    # Context regions preserved for next input.
    assert input_state.text == ""
    assert len(input_state.context_regions) == 1
    assert input_state.context_regions[0].marker == "[/review]"


def test_submit_shell_command_preserves_context(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Enter with context + `!git status` submits it; context survives."""
    input_state.set_text("!git status")
    input_state.add_context("[/review]", "Selected PR.")

    input_reader._on_key(KeyPress(Keys.Enter, "\r"))

    submitted = input_state.submitted.get_nowait()
    assert submitted == "!git status"
    assert len(input_state.context_regions) == 1


def test_submit_regular_text_expands_markers(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Enter with context regions + user text expands normally."""
    input_state.set_text("what changed?")
    input_state.add_context("[/review]", "Selected PR.")

    input_reader._on_key(KeyPress(Keys.Enter, "\r"))

    submitted = input_state.submitted.get_nowait()
    assert "<recent_actions>" in submitted
    assert "what changed?" in submitted
    # Buffer fully cleared after expansion.
    assert input_state.text == ""
    assert input_state.paste_regions == []
    assert input_state.context_regions == []


# ── Backspace dismiss ────────────────────────────────────────────────


def test_backspace_at_pos0_pops_last_context(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Backspace at cursor 0 removes the last context region."""
    input_state.add_context("[/review]", "Selected PR.")
    input_state.add_context("[/model]", "Switched.")

    input_reader._on_key(KeyPress(Keys.Backspace, "\x7f"))

    assert len(input_state.context_regions) == 1
    assert input_state.context_regions[0].marker == "[/review]"


def test_backspace_at_pos0_empties_context(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Backspace at cursor 0 with one region removes it."""
    input_state.add_context("[/review]", "Selected PR.")

    input_reader._on_key(KeyPress(Keys.Backspace, "\x7f"))

    assert input_state.context_regions == []


def test_backspace_at_pos0_noop_when_no_context(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Backspace at cursor 0 with no context is a no-op."""

    input_reader._on_key(KeyPress(Keys.Backspace, "\x7f"))

    assert input_state.text == ""
    assert input_state.context_regions == []


def test_backspace_with_text_does_not_dismiss(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Backspace with cursor inside text deletes a char, not context."""
    input_state.set_text("abc", cursor=3)
    input_state.add_context("[/review]", "Selected PR.")

    input_reader._on_key(KeyPress(Keys.Backspace, "\x7f"))

    assert input_state.text == "ab"
    assert len(input_state.context_regions) == 1
