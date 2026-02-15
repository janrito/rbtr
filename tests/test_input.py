"""Tests for InputState editing via prompt_toolkit Buffer.

Unit tests for the InputState convenience methods and the underlying
Buffer editing — pure state manipulation, no threads or I/O.
"""

import pytest

from rbtr.input import InputState

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def inp():
    """Create a fresh InputState."""
    return InputState()


def _inp(text: str, cursor: int | None = None) -> InputState:
    """Create an InputState with buffer and cursor positioned."""
    state = InputState()
    state.set_text(text, cursor=cursor)
    return state


# ── set_text ─────────────────────────────────────────────────────────


def test_set_text_moves_cursor_to_end():
    s = _inp("")
    s.set_text("hello world")
    assert s.text == "hello world"
    assert s.cursor == 11


def test_set_text_explicit_cursor():
    s = _inp("")
    s.set_text("hello", cursor=3)
    assert s.cursor == 3


def test_set_text_cursor_clamped():
    s = _inp("")
    s.set_text("hi", cursor=99)
    assert s.cursor == 2


# ── insert ───────────────────────────────────────────────────────────


def test_insert_at_end():
    s = _inp("hello")
    s.buffer.insert_text(" world")
    assert s.text == "hello world"
    assert s.cursor == 11


def test_insert_at_start():
    s = _inp("world", cursor=0)
    s.buffer.insert_text("hello ")
    assert s.text == "hello world"
    assert s.cursor == 6


def test_insert_in_middle():
    s = _inp("hllo", cursor=1)
    s.buffer.insert_text("e")
    assert s.text == "hello"
    assert s.cursor == 2


# ── delete_before_cursor (backspace) ─────────────────────────────────


def test_backspace_at_end():
    s = _inp("hello")
    s.buffer.delete_before_cursor()
    assert s.text == "hell"
    assert s.cursor == 4


def test_backspace_in_middle():
    s = _inp("hello", cursor=3)
    s.buffer.delete_before_cursor()
    assert s.text == "helo"
    assert s.cursor == 2


def test_backspace_at_start_noop():
    s = _inp("hello", cursor=0)
    s.buffer.delete_before_cursor()
    assert s.text == "hello"
    assert s.cursor == 0


# ── delete (delete key) ─────────────────────────────────────────────


def test_delete_at_start():
    s = _inp("hello", cursor=0)
    s.buffer.delete()
    assert s.text == "ello"
    assert s.cursor == 0


def test_delete_in_middle():
    s = _inp("hello", cursor=2)
    s.buffer.delete()
    assert s.text == "helo"
    assert s.cursor == 2


def test_delete_at_end_noop():
    s = _inp("hello")
    s.buffer.delete()
    assert s.text == "hello"
    assert s.cursor == 5


# ── word deletion (Ctrl+W) ──────────────────────────────────────────


def test_delete_word_back():
    s = _inp("hello world")
    pos = s.buffer.document.find_previous_word_beginning()
    if pos:
        s.buffer.delete_before_cursor(-pos)
    assert s.text == "hello "
    assert s.cursor == 6


def test_delete_word_back_at_start_noop():
    s = _inp("hello", cursor=0)
    pos = s.buffer.document.find_previous_word_beginning()
    if pos:
        s.buffer.delete_before_cursor(-pos)
    assert s.text == "hello"


# ── word navigation ─────────────────────────────────────────────────


def test_word_left():
    s = _inp("hello world")
    pos = s.buffer.document.find_previous_word_beginning()
    if pos:
        s.buffer.cursor_position += pos
    assert s.cursor == 6


def test_word_right():
    s = _inp("hello world", cursor=0)
    pos = s.buffer.document.find_next_word_ending()
    if pos:
        s.buffer.cursor_position += pos
    assert s.cursor == 5


# ── kill_to_end / kill_to_start ──────────────────────────────────────


def test_kill_to_end():
    s = _inp("hello world", cursor=5)
    end = s.buffer.document.get_end_of_line_position()
    if end > 0:
        s.buffer.delete(end)
    assert s.text == "hello"
    assert s.cursor == 5


def test_kill_to_end_at_end():
    s = _inp("hello")
    end = s.buffer.document.get_end_of_line_position()
    if end > 0:
        s.buffer.delete(end)
    assert s.text == "hello"


def test_kill_to_start():
    s = _inp("hello world", cursor=6)
    col = s.buffer.document.cursor_position_col
    if col > 0:
        s.buffer.delete_before_cursor(col)
    assert s.text == "world"
    assert s.cursor == 0


def test_kill_to_start_at_start():
    s = _inp("hello", cursor=0)
    col = s.buffer.document.cursor_position_col
    if col > 0:
        s.buffer.delete_before_cursor(col)
    assert s.text == "hello"
    assert s.cursor == 0


# ── reset / clear ────────────────────────────────────────────────────


def test_reset_clears_buffer():
    s = _inp("hello world")
    s.reset()
    assert s.text == ""
    assert s.cursor == 0


# ── multiline ────────────────────────────────────────────────────────


def test_multiline_insert_newline():
    s = _inp("line1")
    s.buffer.insert_text("\nline2")
    assert s.text == "line1\nline2"
    assert s.buffer.document.line_count == 2


def test_multiline_cursor_position_row():
    s = _inp("line1\nline2\nline3")
    assert s.buffer.document.cursor_position_row == 2


def test_multiline_cursor_up():
    s = _inp("line1\nline2")
    s.buffer.cursor_up()
    assert s.buffer.document.cursor_position_row == 0


def test_multiline_paste_preserves_newlines():
    s = _inp("")
    s.buffer.insert_text("line1\nline2\nline3")
    assert s.text == "line1\nline2\nline3"
    assert s.buffer.document.line_count == 3


# ── accept_completion ────────────────────────────────────────────────


def test_accept_completion_slash_command():
    s = _inp("/hel")
    s.accept_completion("/help")
    assert s.text == "/help"


def test_accept_completion_slash_with_suffix():
    s = _inp("/hel")
    s.accept_completion("/help", with_suffix=True)
    assert s.text == "/help "


def test_accept_completion_shell_word():
    s = _inp("!git sta")
    s.accept_completion("status")
    assert s.text == "!git status"


def test_accept_completion_shell_with_suffix():
    s = _inp("!git sta")
    s.accept_completion("status", with_suffix=True)
    assert s.text == "!git status "


# ── History ──────────────────────────────────────────────────────────


def test_history_append_adds_to_list(inp):
    inp.append_history("first")
    inp.append_history("second")
    assert inp.history == ["first", "second"]


def test_history_append_writes_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr("rbtr.input.HISTORY_PATH", tmp_path / "history")
    s = InputState()
    s.append_history("/review")
    s.append_history("!git status")
    assert (tmp_path / "history").read_text() == "/review\n!git status\n"


def test_history_append_bounds_at_500():
    s = InputState()
    s.history = [f"cmd{i}" for i in range(500)]
    s.append_history("new")
    assert len(s.history) == 500
    assert s.history[-1] == "new"
    assert s.history[0] == "cmd1"


# ── History prefix search ───────────────────────────────────────────


def _make_reader(history: list[str]):
    """Create a bare InputReader with just enough state for history tests."""
    from rbtr.input import InputReader

    state = InputState()
    state.history = list(history)
    reader = object.__new__(InputReader)
    reader._state = state
    reader._history_index = -1
    reader._saved_text = ""
    reader._search_prefix = ""
    return reader, state


def test_history_up_cycles_all_when_empty():
    reader, state = _make_reader(["a", "b", "c"])
    reader._history_up()
    assert state.text == "c"
    reader._history_up()
    assert state.text == "b"
    reader._history_up()
    assert state.text == "a"


def test_history_up_filters_by_prefix():
    reader, state = _make_reader(["!git status", "/review", "!git log", "hello"])
    state.set_text("!git")
    reader._history_up()
    assert state.text == "!git log"
    reader._history_up()
    assert state.text == "!git status"


def test_history_up_stops_at_top():
    reader, state = _make_reader(["!git status", "/review"])
    state.set_text("!git")
    reader._history_up()
    assert state.text == "!git status"
    reader._history_up()
    assert state.text == "!git status"


def test_history_down_restores_original():
    reader, state = _make_reader(["abc", "abd", "xyz"])
    state.set_text("ab")
    reader._history_up()
    assert state.text == "abd"
    reader._history_down()
    assert state.text == "ab"


def test_history_down_filters_forward():
    reader, state = _make_reader(["!git status", "/review", "!git log"])
    state.set_text("!git")
    reader._history_up()   # → !git log
    reader._history_up()   # → !git status
    reader._history_down()  # → !git log
    assert state.text == "!git log"


def test_history_prefix_locked_on_first_up():
    """Prefix is saved from original text, not from history entries."""
    reader, state = _make_reader(["!git status", "!git log --oneline"])
    state.set_text("!git l")
    reader._history_up()
    assert state.text == "!git log --oneline"
    reader._history_up()
    assert state.text == "!git log --oneline"
