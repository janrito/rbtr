"""Tests for InputState editing via prompt_toolkit Buffer.

Unit tests for the InputState convenience methods and the underlying
Buffer editing — pure state manipulation, no threads or I/O.
"""

import pytest
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys

from rbtr.config import config
from rbtr.tui.input import InputReader, InputState, PasteRegion, make_paste_marker

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def inp() -> InputState:
    """Create a fresh InputState."""
    return InputState()


def _inp(text: str, cursor: int | None = None) -> InputState:
    """Create an InputState with buffer and cursor positioned."""
    state = InputState()
    state.set_text(text, cursor=cursor)
    return state


# ── set_text ─────────────────────────────────────────────────────────


def test_set_text_moves_cursor_to_end() -> None:
    s = _inp("")
    s.set_text("hello world")
    assert s.text == "hello world"
    assert s.cursor == 11


def test_set_text_explicit_cursor() -> None:
    s = _inp("")
    s.set_text("hello", cursor=3)
    assert s.cursor == 3


def test_set_text_cursor_clamped() -> None:
    s = _inp("")
    s.set_text("hi", cursor=99)
    assert s.cursor == 2


# ── insert ───────────────────────────────────────────────────────────


def test_insert_at_end() -> None:
    s = _inp("hello")
    s.buffer.insert_text(" world")
    assert s.text == "hello world"
    assert s.cursor == 11


def test_insert_at_start() -> None:
    s = _inp("world", cursor=0)
    s.buffer.insert_text("hello ")
    assert s.text == "hello world"
    assert s.cursor == 6


def test_insert_in_middle() -> None:
    s = _inp("hllo", cursor=1)
    s.buffer.insert_text("e")
    assert s.text == "hello"
    assert s.cursor == 2


# ── delete_before_cursor (backspace) ─────────────────────────────────


def test_backspace_at_end() -> None:
    s = _inp("hello")
    s.buffer.delete_before_cursor()
    assert s.text == "hell"
    assert s.cursor == 4


def test_backspace_in_middle() -> None:
    s = _inp("hello", cursor=3)
    s.buffer.delete_before_cursor()
    assert s.text == "helo"
    assert s.cursor == 2


def test_backspace_at_start_noop() -> None:
    s = _inp("hello", cursor=0)
    s.buffer.delete_before_cursor()
    assert s.text == "hello"
    assert s.cursor == 0


# ── delete (delete key) ─────────────────────────────────────────────


def test_delete_at_start() -> None:
    s = _inp("hello", cursor=0)
    s.buffer.delete()
    assert s.text == "ello"
    assert s.cursor == 0


def test_delete_in_middle() -> None:
    s = _inp("hello", cursor=2)
    s.buffer.delete()
    assert s.text == "helo"
    assert s.cursor == 2


def test_delete_at_end_noop() -> None:
    s = _inp("hello")
    s.buffer.delete()
    assert s.text == "hello"
    assert s.cursor == 5


# ── word deletion (Ctrl+W) ──────────────────────────────────────────


def test_delete_word_back() -> None:
    s = _inp("hello world")
    pos = s.buffer.document.find_previous_word_beginning()
    if pos:
        s.buffer.delete_before_cursor(-pos)
    assert s.text == "hello "
    assert s.cursor == 6


def test_delete_word_back_at_start_noop() -> None:
    s = _inp("hello", cursor=0)
    pos = s.buffer.document.find_previous_word_beginning()
    if pos:
        s.buffer.delete_before_cursor(-pos)
    assert s.text == "hello"


# ── word navigation ─────────────────────────────────────────────────


def test_word_left() -> None:
    s = _inp("hello world")
    pos = s.buffer.document.find_previous_word_beginning()
    if pos:
        s.buffer.cursor_position += pos
    assert s.cursor == 6


def test_word_right() -> None:
    s = _inp("hello world", cursor=0)
    pos = s.buffer.document.find_next_word_ending()
    if pos:
        s.buffer.cursor_position += pos
    assert s.cursor == 5


# ── kill_to_end / kill_to_start ──────────────────────────────────────


def test_kill_to_end() -> None:
    s = _inp("hello world", cursor=5)
    end = s.buffer.document.get_end_of_line_position()
    if end > 0:
        s.buffer.delete(end)
    assert s.text == "hello"
    assert s.cursor == 5


def test_kill_to_end_at_end() -> None:
    s = _inp("hello")
    end = s.buffer.document.get_end_of_line_position()
    if end > 0:
        s.buffer.delete(end)
    assert s.text == "hello"


def test_kill_to_start() -> None:
    s = _inp("hello world", cursor=6)
    col = s.buffer.document.cursor_position_col
    if col > 0:
        s.buffer.delete_before_cursor(col)
    assert s.text == "world"
    assert s.cursor == 0


def test_kill_to_start_at_start() -> None:
    s = _inp("hello", cursor=0)
    col = s.buffer.document.cursor_position_col
    if col > 0:
        s.buffer.delete_before_cursor(col)
    assert s.text == "hello"
    assert s.cursor == 0


# ── reset / clear ────────────────────────────────────────────────────


def test_reset_clears_buffer() -> None:
    s = _inp("hello world")
    s.reset()
    assert s.text == ""
    assert s.cursor == 0


# ── multiline ────────────────────────────────────────────────────────


def test_multiline_insert_newline() -> None:
    s = _inp("line1")
    s.buffer.insert_text("\nline2")
    assert s.text == "line1\nline2"
    assert s.buffer.document.line_count == 2


def test_multiline_cursor_position_row() -> None:
    s = _inp("line1\nline2\nline3")
    assert s.buffer.document.cursor_position_row == 2


def test_multiline_cursor_up() -> None:
    s = _inp("line1\nline2")
    s.buffer.cursor_up()
    assert s.buffer.document.cursor_position_row == 0


def test_multiline_paste_preserves_newlines() -> None:
    s = _inp("")
    s.buffer.insert_text("line1\nline2\nline3")
    assert s.text == "line1\nline2\nline3"
    assert s.buffer.document.line_count == 3


# ── accept_completion ────────────────────────────────────────────────


def test_accept_completion_slash_command() -> None:
    s = _inp("/hel")
    s.accept_completion("/help")
    assert s.text == "/help"


def test_accept_completion_slash_with_suffix() -> None:
    s = _inp("/hel")
    s.accept_completion("/help", with_suffix=True)
    assert s.text == "/help "


def test_accept_completion_shell_word() -> None:
    s = _inp("!git sta")
    s.accept_completion("status")
    assert s.text == "!git status"


def test_accept_completion_shell_with_suffix() -> None:
    s = _inp("!git sta")
    s.accept_completion("status", with_suffix=True)
    assert s.text == "!git status "


# ── History ──────────────────────────────────────────────────────────


def test_history_append_adds_to_list(inp: InputState) -> None:
    inp.append_history("first")
    inp.append_history("second")
    assert inp.history == ["first", "second"]


def test_history_append_no_disk_write() -> None:
    """append_history updates in-memory list only — no disk I/O."""
    s = InputState()
    s.append_history("/review")
    s.append_history("!git status")
    assert s.history == ["/review", "!git status"]


def test_history_append_bounds_at_max() -> None:
    """In-memory history is capped at config.tui.max_history."""

    limit = config.tui.max_history
    s = InputState()
    s.history = [f"cmd{i}" for i in range(limit)]
    s.append_history("new")
    assert len(s.history) == limit
    assert s.history[-1] == "new"
    assert s.history[0] == "cmd1"


# ── History prefix search ───────────────────────────────────────────


def _make_reader(history: list[str]) -> tuple[InputReader, InputState]:
    """Create a bare InputReader with just enough state for history tests."""

    state = InputState()
    state.history = list(history)
    reader = object.__new__(InputReader)
    reader._state = state
    reader._history_index = -1
    reader._saved_text = ""
    reader._search_prefix = ""
    return reader, state


def test_history_up_cycles_all_when_empty() -> None:
    reader, state = _make_reader(["a", "b", "c"])
    reader._history_up()
    assert state.text == "c"
    reader._history_up()
    assert state.text == "b"
    reader._history_up()
    assert state.text == "a"


def test_history_up_filters_by_prefix() -> None:
    reader, state = _make_reader(["!git status", "/review", "!git log", "hello"])
    state.set_text("!git")
    reader._history_up()
    assert state.text == "!git log"
    reader._history_up()
    assert state.text == "!git status"


def test_history_up_stops_at_top() -> None:
    reader, state = _make_reader(["!git status", "/review"])
    state.set_text("!git")
    reader._history_up()
    assert state.text == "!git status"
    reader._history_up()
    assert state.text == "!git status"


def test_history_down_restores_original() -> None:
    reader, state = _make_reader(["abc", "abd", "xyz"])
    state.set_text("ab")
    reader._history_up()
    assert state.text == "abd"
    reader._history_down()
    assert state.text == "ab"


def test_history_down_filters_forward() -> None:
    reader, state = _make_reader(["!git status", "/review", "!git log"])
    state.set_text("!git")
    reader._history_up()  # → !git log
    reader._history_up()  # → !git status
    reader._history_down()  # → !git log
    assert state.text == "!git log"


def test_history_prefix_locked_on_first_up() -> None:
    """Prefix is saved from original text, not from history entries."""
    reader, state = _make_reader(["!git status", "!git log --oneline"])
    state.set_text("!git l")
    reader._history_up()
    assert state.text == "!git log --oneline"
    reader._history_up()
    assert state.text == "!git log --oneline"


# ── History provider (DB-backed) ────────────────────────────────────


def _make_reader_with_provider(provider_data: list[str]) -> tuple[InputReader, InputState]:
    """Create a reader with a history provider."""

    # Provider returns most-recent-first (like search_history).
    def provider(prefix: str | None, limit: int) -> list[str]:
        results = provider_data
        if prefix is not None:
            results = [r for r in results if r.startswith(prefix)]
        return results[:limit]

    state = InputState(history_provider=provider)
    reader = object.__new__(InputReader)
    reader._state = state
    reader._history_index = -1
    reader._saved_text = ""
    reader._search_prefix = ""

    return reader, state


def test_provider_seeds_history() -> None:
    """History provider seeds the in-memory list (reversed to chronological)."""
    reader, state = _make_reader_with_provider(["newest", "middle", "oldest"])
    reader._load_history()
    assert state.history == ["oldest", "middle", "newest"]


def test_provider_empty_leaves_history_empty() -> None:
    """When provider returns empty, history stays empty."""
    reader, state = _make_reader_with_provider([])
    reader._load_history()
    assert state.history == []


def test_no_provider_leaves_history_empty() -> None:
    """Without a provider, history stays empty (no legacy fallback)."""

    state = InputState()
    reader = object.__new__(InputReader)
    reader._state = state
    reader._history_index = -1
    reader._saved_text = ""
    reader._search_prefix = ""
    reader._load_history()
    assert state.history == []


# ── Paste markers ────────────────────────────────────────────────────


def _make_key_reader(state: InputState | None = None) -> tuple[InputReader, InputState]:
    """Create a bare InputReader wired to *state* for key dispatch tests."""
    if state is None:
        state = InputState()
    reader = object.__new__(InputReader)
    reader._state = state
    reader._escape_next = False
    reader._history_index = -1
    reader._saved_text = ""
    reader._search_prefix = ""
    return reader, state


def _send(reader: InputReader, key: Keys | str, data: str = "") -> None:
    """Feed a single key press through the reader's dispatch."""
    reader._on_key(KeyPress(key, data))


# ── make_paste_marker ────────────────────────────────────────────────


def test_make_marker_multiline() -> None:
    assert make_paste_marker("a\nb\nc\nd\n", []) == "[pasted 5 lines]"


def test_make_marker_single_line() -> None:
    assert make_paste_marker("x" * 300, []) == "[pasted 300 chars]"


def test_make_marker_disambiguates_duplicates() -> None:
    existing = [PasteRegion(marker="[pasted 5 lines]", content="a\nb\nc\nd\ne")]
    assert make_paste_marker("1\n2\n3\n4\n5", existing) == "[pasted 5 lines #2]"


def test_make_marker_triple_duplicate() -> None:
    existing = [
        PasteRegion(marker="[pasted 5 lines]", content="a"),
        PasteRegion(marker="[pasted 5 lines #2]", content="b"),
    ]
    assert make_paste_marker("1\n2\n3\n4\n5", existing) == "[pasted 5 lines #3]"


# ── Paste via key dispatch ───────────────────────────────────────────


def test_paste_below_threshold_inserts_inline() -> None:
    """Short paste (< 4 lines) inserts text directly, no marker."""
    reader, state = _make_key_reader()
    _send(reader, Keys.BracketedPaste, "line1\nline2")
    assert state.text == "line1\nline2"
    assert state.paste_regions == []


def test_paste_above_line_threshold_creates_marker() -> None:
    """Paste with ≥ 4 lines collapses to a marker in the buffer."""
    reader, state = _make_key_reader()
    content = "line1\nline2\nline3\nline4"
    _send(reader, Keys.BracketedPaste, content)
    assert state.text == "[pasted 4 lines]"
    assert len(state.paste_regions) == 1
    assert state.paste_regions[0].content == content


def test_paste_above_char_threshold_creates_marker() -> None:
    """Long single-line paste (> 200 chars) collapses to a char marker."""
    reader, state = _make_key_reader()
    content = "x" * 250
    _send(reader, Keys.BracketedPaste, content)
    assert state.text == "[pasted 250 chars]"
    assert state.paste_regions[0].content == content


def test_paste_at_cursor_position() -> None:
    """Marker is inserted at cursor position, not appended."""
    reader, state = _make_key_reader()
    state.set_text("hello  world", cursor=6)
    _send(reader, Keys.BracketedPaste, "a\nb\nc\nd")
    assert state.text == "hello [pasted 4 lines] world"


# ── Submit expands markers ───────────────────────────────────────────


def test_submit_expands_markers() -> None:
    """Enter submits the real pasted content, not the marker text."""
    reader, state = _make_key_reader()
    content = "line1\nline2\nline3\nline4"
    _send(reader, Keys.BracketedPaste, content)
    _send(reader, Keys.Enter, "\r")
    submitted = state.submitted.get_nowait()
    assert submitted == content


def test_submit_expands_mixed_text_and_markers() -> None:
    """Text typed around a marker is preserved in the submitted string."""
    reader, state = _make_key_reader()
    state.set_text("review this: ")
    content = "a\nb\nc\nd"
    _send(reader, Keys.BracketedPaste, content)
    _send(reader, Keys.Enter, "\r")
    submitted = state.submitted.get_nowait()
    assert submitted == f"review this: {content}"


# ── expand_markers / marker_span_at (pure helpers) ───────────────────


def test_expand_markers_no_regions() -> None:
    s = _inp("no markers here")
    assert s.expand_markers(s.text) == "no markers here"


def test_expand_markers_multiple_regions() -> None:
    s = _inp("[pasted 4 lines] and [pasted 200 chars]")
    s.paste_regions.append(PasteRegion(marker="[pasted 4 lines]", content="a\nb\nc\nd"))
    s.paste_regions.append(PasteRegion(marker="[pasted 200 chars]", content="z" * 200))
    assert s.expand_markers(s.text) == "a\nb\nc\nd and " + "z" * 200


def test_marker_span_at_inside_returns_span_and_region() -> None:
    s = _inp("abc[pasted 5 lines]xyz")
    region = PasteRegion(marker="[pasted 5 lines]", content="...")
    s.paste_regions.append(region)
    span = s.marker_span_at(3)
    assert span is not None
    assert span[:2] == (3, 19)
    assert span[2] is region
    # Last char inside marker (pos 18, exclusive end 19).
    assert s.marker_span_at(18) is not None


def test_marker_span_at_outside_returns_none() -> None:
    s = _inp("abc[pasted 5 lines]xyz")
    s.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="..."))
    assert s.marker_span_at(2) is None
    assert s.marker_span_at(19) is None


# ── remove_marker ────────────────────────────────────────────────────


def test_remove_marker_deletes_text_and_region() -> None:
    s = _inp("abc[pasted 5 lines]xyz")
    region = PasteRegion(marker="[pasted 5 lines]", content="real")
    s.paste_regions.append(region)
    s.remove_marker(region)
    assert s.text == "abcxyz"
    assert s.paste_regions == []


def test_remove_marker_cursor_after_marker_shifts_left() -> None:
    s = _inp("abc[pasted 5 lines]xyz", cursor=19)
    region = PasteRegion(marker="[pasted 5 lines]", content="real")
    s.paste_regions.append(region)
    s.remove_marker(region)
    assert s.cursor == 3


def test_remove_marker_cursor_inside_marker_snaps_to_start() -> None:
    """Cursor inside marker (defensive) collapses to marker start."""
    s = _inp("abc[pasted 5 lines]xyz", cursor=10)
    region = PasteRegion(marker="[pasted 5 lines]", content="real")
    s.paste_regions.append(region)
    s.remove_marker(region)
    assert s.text == "abcxyz"
    assert s.cursor == 3


def test_remove_marker_cursor_before_marker_unchanged() -> None:
    s = _inp("abc[pasted 5 lines]xyz", cursor=1)
    region = PasteRegion(marker="[pasted 5 lines]", content="real")
    s.paste_regions.append(region)
    s.remove_marker(region)
    assert s.cursor == 1


# ── prune_orphaned_regions ───────────────────────────────────────────


def test_prune_removes_orphaned_regions() -> None:
    s = _inp("only [pasted 5 lines] remains")
    s.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="a"))
    s.paste_regions.append(PasteRegion(marker="[pasted 3 lines]", content="b"))
    s.prune_orphaned_regions()
    assert len(s.paste_regions) == 1
    assert s.paste_regions[0].marker == "[pasted 5 lines]"


# ── reset clears regions ─────────────────────────────────────────────


def test_reset_clears_paste_regions() -> None:
    s = _inp("[pasted 5 lines]")
    s.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="a"))
    s.reset()
    assert s.text == ""
    assert s.paste_regions == []


# ── Atomic cursor navigation via key dispatch ────────────────────────


def test_key_left_snaps_past_marker() -> None:
    """Left arrow one step into a marker snaps cursor to marker start."""
    state = _inp("abc[pasted 5 lines]xyz", cursor=19)
    state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="..."))
    reader, _ = _make_key_reader(state)
    _send(reader, Keys.Left, "")
    assert state.cursor == 3


def test_key_right_snaps_past_marker() -> None:
    """Right arrow at marker start snaps cursor to marker end."""
    state = _inp("abc[pasted 5 lines]xyz", cursor=3)
    state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="..."))
    reader, _ = _make_key_reader(state)
    _send(reader, Keys.Right, "")
    assert state.cursor == 19


# ── Atomic deletion via key dispatch ─────────────────────────────────


def test_key_backspace_at_marker_end_removes_marker() -> None:
    """Backspace right after a marker removes the entire marker + region."""
    state = _inp("abc[pasted 5 lines]xyz", cursor=19)
    state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="real"))
    reader, _ = _make_key_reader(state)
    _send(reader, Keys.Backspace, "\x7f")
    assert state.text == "abcxyz"
    assert state.paste_regions == []
    assert state.cursor == 3


def test_key_delete_at_marker_start_removes_marker() -> None:
    """Delete key at marker start removes the entire marker + region."""
    state = _inp("abc[pasted 5 lines]xyz", cursor=3)
    state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="real"))
    reader, _ = _make_key_reader(state)
    _send(reader, Keys.Delete, "")
    assert state.text == "abcxyz"
    assert state.paste_regions == []


def test_key_ctrl_w_removes_overlapping_marker() -> None:
    """Ctrl+W with cursor after marker removes the marker atomically."""
    state = _inp("hello [pasted 4 lines]", cursor=22)
    state.paste_regions.append(PasteRegion(marker="[pasted 4 lines]", content="a\nb\nc\nd"))
    reader, _ = _make_key_reader(state)
    _send(reader, Keys.ControlW, "")
    # Ctrl+W kills back to word boundary; the marker overlaps → whole marker gone.
    assert "[pasted" not in state.text
    assert state.paste_regions == []


def test_key_ctrl_u_removes_overlapping_marker() -> None:
    """Ctrl+U kills to line start, removing any marker in the range."""
    state = _inp("abc[pasted 4 lines]xyz", cursor=22)
    state.paste_regions.append(PasteRegion(marker="[pasted 4 lines]", content="data"))
    reader, _ = _make_key_reader(state)
    _send(reader, Keys.ControlU, "")
    assert state.paste_regions == []


# ── Multiple pastes ──────────────────────────────────────────────────


def test_multiple_pastes_expand_independently() -> None:
    """Two pastes produce two markers; expand replaces both."""
    reader, state = _make_key_reader()
    _send(reader, Keys.BracketedPaste, "a\nb\nc\nd")
    state.buffer.insert_text(" ")
    _send(reader, Keys.BracketedPaste, "e\nf\ng\nh")
    assert len(state.paste_regions) == 2
    expanded = state.expand_markers(state.text)
    assert expanded == "a\nb\nc\nd e\nf\ng\nh"


def test_multiple_pastes_delete_one_preserves_other() -> None:
    """Deleting one marker preserves the other."""
    state = _inp("[pasted 4 lines] [pasted 4 lines #2]")
    r1 = PasteRegion(marker="[pasted 4 lines]", content="first")
    r2 = PasteRegion(marker="[pasted 4 lines #2]", content="second")
    state.paste_regions.extend([r1, r2])
    state.remove_marker(r1)
    assert "[pasted 4 lines #2]" in state.text
    assert len(state.paste_regions) == 1
    assert state.paste_regions[0].content == "second"
