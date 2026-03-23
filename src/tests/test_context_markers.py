"""Tests for context markers -- creation, insertion, cursor, expansion, and shell.

Covers Phases 1-5 of the context-marker feature:
- `MarkerKind` enum and `PasteRegion` with `kind=CONTEXT`
- Context marker insertion into the input buffer
- Cursor preservation when markers are inserted
- Expansion format (single, multiple, mixed, absent)
- Shell context markers (success, failure, truncation)
"""

from __future__ import annotations

import queue
from collections.abc import Generator

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from rbtr.config import config
from rbtr.engine.core import Engine
from rbtr.engine.shell import _emit_shell_context
from rbtr.engine.types import TaskType
from rbtr.events import ContextMarkerReady, Event
from rbtr.sessions.store import SessionStore
from rbtr.shell_exec import ShellResult
from rbtr.state import EngineState
from rbtr.tui.input import InputState, MarkerKind, PasteRegion
from tests.helpers import drain

# ── Helpers ──────────────────────────────────────────────────────────


def _inp(text: str = "", cursor: int | None = None) -> InputState:
    state = InputState()
    state.set_text(text, cursor=cursor)
    return state


def _context_region(marker: str, content: str) -> PasteRegion:
    return PasteRegion(marker=marker, content=content, kind=MarkerKind.CONTEXT)


def _paste_region(marker: str, content: str) -> PasteRegion:
    return PasteRegion(marker=marker, content=content, kind=MarkerKind.PASTE)


def _insert_context_marker(state: InputState, marker: str, content: str) -> None:
    """Replicate the TUI's `_insert_context_marker` logic."""
    region = PasteRegion(marker=marker, content=content, kind=MarkerKind.CONTEXT)
    state.paste_regions.append(region)
    insert_pos = 0
    for _s, span_end, span_region in state.marker_spans():
        if span_region.kind is MarkerKind.CONTEXT:
            insert_pos = max(insert_pos, span_end)
    text = state.text
    new_text = text[:insert_pos] + marker + " " + text[insert_pos:]
    old_cursor = state.cursor
    new_cursor = old_cursor + len(marker) + 1 if old_cursor >= insert_pos else old_cursor
    state.set_text(new_text, cursor=new_cursor)


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


# ── MarkerKind and PasteRegion with kind=CONTEXT ─────────────────────


def test_marker_kind_values() -> None:
    assert MarkerKind.PASTE == "paste"
    assert MarkerKind.CONTEXT == "context"


def test_paste_region_default_kind_is_paste() -> None:
    r = PasteRegion(marker="[pasted 5 lines]", content="data")
    assert r.kind is MarkerKind.PASTE


def test_paste_region_context_kind() -> None:
    r = _context_region("[/review → PR #42]", "Selected PR #42.")
    assert r.kind is MarkerKind.CONTEXT
    assert r.marker == "[/review → PR #42]"
    assert r.content == "Selected PR #42."


def test_context_region_detected_in_spans() -> None:
    marker = "[/model gpt-4o]"
    state = _inp(f"{marker} hello")
    state.paste_regions.append(_context_region(marker, "Switched."))
    spans = state.marker_spans()
    assert len(spans) == 1
    start, end, region = spans[0]
    assert start == 0
    assert end == len(marker)
    assert region.kind is MarkerKind.CONTEXT


def test_context_region_deletion() -> None:
    marker = "[/model gpt-4o]"
    state = _inp(f"{marker} hello")
    region = _context_region(marker, "Switched.")
    state.paste_regions.append(region)
    state.remove_marker(region)
    assert state.text == " hello"
    assert state.paste_regions == []


# ── Context marker insertion into input buffer ───────────────────────


def test_insert_into_empty_buffer() -> None:
    state = _inp("")
    _insert_context_marker(state, "[/review → PR #1]", "Selected PR.")
    assert state.text == "[/review → PR #1] "
    assert len(state.paste_regions) == 1


def test_insert_with_existing_text() -> None:
    state = _inp("hello world", cursor=5)
    _insert_context_marker(state, "[/model → gpt-4o]", "Switched.")
    assert state.text == "[/model → gpt-4o] hello world"
    # Cursor shifted right by marker+space.
    assert state.cursor == 5 + len("[/model → gpt-4o]") + 1


def test_multiple_markers_in_order() -> None:
    state = _inp("")
    _insert_context_marker(state, "[/review → PR #1]", "PR selected.")
    _insert_context_marker(state, "[/model → gpt-4o]", "Switched.")
    # Second marker appears after first (chronological order).
    assert "[/review → PR #1]" in state.text
    assert "[/model → gpt-4o]" in state.text
    idx_r = state.text.find("[/review → PR #1]")
    idx_m = state.text.find("[/model → gpt-4o]")
    assert idx_r < idx_m


def test_cursor_preserved_during_typing() -> None:
    """User is typing at position 5; marker inserts at 0 without disruption."""
    state = _inp("hello world", cursor=5)
    _insert_context_marker(state, "[/review → PR #1]", "Selected.")
    expected = 5 + len("[/review → PR #1]") + 1
    assert state.cursor == expected
    assert state.text.endswith("hello world")


# ── Expansion format on submit ───────────────────────────────────────


def test_expand_no_markers_no_prefix() -> None:
    state = _inp("just a question")
    assert state.expand_markers(state.text) == "just a question"


def test_expand_single_context_marker() -> None:
    state = _inp("[/review → PR #42] what changed?")
    state.paste_regions.append(_context_region("[/review → PR #42]", "Selected PR #42."))
    result = state.expand_markers(state.text)
    assert result == "[Recent actions]\n- Selected PR #42.\n\n---\nwhat changed?"


def test_expand_multiple_context_markers() -> None:
    state = _inp("[/review → PR #42] [/model → gpt-4o] summarise")
    state.paste_regions.extend(
        [
            _context_region("[/review → PR #42]", "Selected PR #42."),
            _context_region("[/model → gpt-4o]", "Switched model."),
        ]
    )
    result = state.expand_markers(state.text)
    assert "[Recent actions]" in result
    assert "- Selected PR #42." in result
    assert "- Switched model." in result
    assert result.endswith("---\nsummarise")


def test_expand_context_markers_only_no_user_text() -> None:
    state = _inp("[/review → PR #42] ")
    state.paste_regions.append(_context_region("[/review → PR #42]", "Selected PR #42."))
    result = state.expand_markers(state.text)
    assert result == "[Recent actions]\n- Selected PR #42."


def test_expand_mixed_paste_and_context() -> None:
    state = _inp("[/review → PR #42] see [pasted 4 lines] above")
    state.paste_regions.extend(
        [
            _context_region("[/review → PR #42]", "Selected PR #42."),
            _paste_region("[pasted 4 lines]", "a\nb\nc\nd"),
        ]
    )
    result = state.expand_markers(state.text)
    assert result.startswith("[Recent actions]\n- Selected PR #42.\n\n---\n")
    assert "a\nb\nc\nd" in result
    assert "[pasted 4 lines]" not in result


def test_expand_user_deletes_all_context_markers() -> None:
    """If user deletes all context markers, message has no prefix."""
    state = _inp("just a question")
    assert state.expand_markers(state.text) == "just a question"


def test_expand_context_order_preserved() -> None:
    """Context entries appear in buffer order (left to right)."""
    state = _inp("[B] [A] question")
    state.paste_regions.extend(
        [
            _context_region("[B]", "Second command."),
            _context_region("[A]", "First command."),
        ]
    )
    result = state.expand_markers(state.text)
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


def test_expanded_message_round_trips() -> None:
    """Expanded context in user messages round-trips through the store."""

    state = _inp("[/review → PR #42] what changed?")
    state.paste_regions.append(_context_region("[/review → PR #42]", "Selected PR #42."))
    text = state.expand_markers(state.text)

    with SessionStore() as store:
        sid = "test-session"
        store.set_context(sid)
        msg = ModelRequest(parts=[UserPromptPart(content=text)])
        store.save_messages(sid, [msg])
        loaded = store.load_messages(sid)

        assert len(loaded) == 1
        assert loaded[0].parts[0].content == text  # type: ignore[union-attr]  # UserPromptPart.content
