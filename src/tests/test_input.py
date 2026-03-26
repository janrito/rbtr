"""Tests for InputState editing via prompt_toolkit Buffer.

Unit tests for the InputState convenience methods and the underlying
Buffer editing — pure state manipulation, no threads or I/O.
"""

from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys

from rbtr.config import config
from rbtr.tui.input import ContextRegion, InputReader, InputState, PasteRegion, make_paste_marker

# ── set_text ─────────────────────────────────────────────────────────


def test_set_text_moves_cursor_to_end(input_state: InputState) -> None:
    input_state.set_text("hello world")
    assert input_state.text == "hello world"
    assert input_state.cursor == 11


def test_set_text_explicit_cursor(input_state: InputState) -> None:
    input_state.set_text("hello", cursor=3)
    assert input_state.cursor == 3


def test_set_text_cursor_clamped(input_state: InputState) -> None:
    input_state.set_text("hi", cursor=99)
    assert input_state.cursor == 2


# ── insert ───────────────────────────────────────────────────────────


def test_insert_at_end(input_state: InputState) -> None:
    input_state.set_text("hello")
    input_state.buffer.insert_text(" world")
    assert input_state.text == "hello world"
    assert input_state.cursor == 11


def test_insert_at_start(input_state: InputState) -> None:
    input_state.set_text("world", cursor=0)
    input_state.buffer.insert_text("hello ")
    assert input_state.text == "hello world"
    assert input_state.cursor == 6


def test_insert_in_middle(input_state: InputState) -> None:
    input_state.set_text("hllo", cursor=1)
    input_state.buffer.insert_text("e")
    assert input_state.text == "hello"
    assert input_state.cursor == 2


# ── delete_before_cursor (backspace) ─────────────────────────────────


def test_backspace_at_end(input_state: InputState) -> None:
    input_state.set_text("hello")
    input_state.buffer.delete_before_cursor()
    assert input_state.text == "hell"
    assert input_state.cursor == 4


def test_backspace_in_middle(input_state: InputState) -> None:
    input_state.set_text("hello", cursor=3)
    input_state.buffer.delete_before_cursor()
    assert input_state.text == "helo"
    assert input_state.cursor == 2


def test_backspace_at_start_noop(input_state: InputState) -> None:
    input_state.set_text("hello", cursor=0)
    input_state.buffer.delete_before_cursor()
    assert input_state.text == "hello"
    assert input_state.cursor == 0


# ── delete (delete key) ─────────────────────────────────────────────


def test_delete_at_start(input_state: InputState) -> None:
    input_state.set_text("hello", cursor=0)
    input_state.buffer.delete()
    assert input_state.text == "ello"
    assert input_state.cursor == 0


def test_delete_in_middle(input_state: InputState) -> None:
    input_state.set_text("hello", cursor=2)
    input_state.buffer.delete()
    assert input_state.text == "helo"
    assert input_state.cursor == 2


def test_delete_at_end_noop(input_state: InputState) -> None:
    input_state.set_text("hello")
    input_state.buffer.delete()
    assert input_state.text == "hello"
    assert input_state.cursor == 5


# ── word deletion (Ctrl+W) ──────────────────────────────────────────


def test_delete_word_back(input_state: InputState) -> None:
    input_state.set_text("hello world")
    pos = input_state.buffer.document.find_previous_word_beginning()
    if pos:
        input_state.buffer.delete_before_cursor(-pos)
    assert input_state.text == "hello "
    assert input_state.cursor == 6


def test_delete_word_back_at_start_noop(input_state: InputState) -> None:
    input_state.set_text("hello", cursor=0)
    pos = input_state.buffer.document.find_previous_word_beginning()
    if pos:
        input_state.buffer.delete_before_cursor(-pos)
    assert input_state.text == "hello"


# ── word navigation ─────────────────────────────────────────────────


def test_word_left(input_state: InputState) -> None:
    input_state.set_text("hello world")
    pos = input_state.buffer.document.find_previous_word_beginning()
    if pos:
        input_state.buffer.cursor_position += pos
    assert input_state.cursor == 6


def test_word_right(input_state: InputState) -> None:
    input_state.set_text("hello world", cursor=0)
    pos = input_state.buffer.document.find_next_word_ending()
    if pos:
        input_state.buffer.cursor_position += pos
    assert input_state.cursor == 5


# ── kill_to_end / kill_to_start ──────────────────────────────────────


def test_kill_to_end(input_state: InputState) -> None:
    input_state.set_text("hello world", cursor=5)
    end = input_state.buffer.document.get_end_of_line_position()
    if end > 0:
        input_state.buffer.delete(end)
    assert input_state.text == "hello"
    assert input_state.cursor == 5


def test_kill_to_end_at_end(input_state: InputState) -> None:
    input_state.set_text("hello")
    end = input_state.buffer.document.get_end_of_line_position()
    if end > 0:
        input_state.buffer.delete(end)
    assert input_state.text == "hello"


def test_kill_to_start(input_state: InputState) -> None:
    input_state.set_text("hello world", cursor=6)
    col = input_state.buffer.document.cursor_position_col
    if col > 0:
        input_state.buffer.delete_before_cursor(col)
    assert input_state.text == "world"
    assert input_state.cursor == 0


def test_kill_to_start_at_start(input_state: InputState) -> None:
    input_state.set_text("hello", cursor=0)
    col = input_state.buffer.document.cursor_position_col
    if col > 0:
        input_state.buffer.delete_before_cursor(col)
    assert input_state.text == "hello"
    assert input_state.cursor == 0


# ── reset / clear ────────────────────────────────────────────────────


def test_reset_clears_buffer(input_state: InputState) -> None:
    input_state.set_text("hello world")
    input_state.reset()
    assert input_state.text == ""
    assert input_state.cursor == 0


# ── multiline ────────────────────────────────────────────────────────


def test_multiline_insert_newline(input_state: InputState) -> None:
    input_state.set_text("line1")
    input_state.buffer.insert_text("\nline2")
    assert input_state.text == "line1\nline2"
    assert input_state.buffer.document.line_count == 2


def test_multiline_cursor_position_row(input_state: InputState) -> None:
    input_state.set_text("line1\nline2\nline3")
    assert input_state.buffer.document.cursor_position_row == 2


def test_multiline_cursor_up(input_state: InputState) -> None:
    input_state.set_text("line1\nline2")
    input_state.buffer.cursor_up()
    assert input_state.buffer.document.cursor_position_row == 0


def test_multiline_paste_preserves_newlines(input_state: InputState) -> None:
    input_state.buffer.insert_text("line1\nline2\nline3")
    assert input_state.text == "line1\nline2\nline3"
    assert input_state.buffer.document.line_count == 3


# ── accept_completion ────────────────────────────────────────────────


def test_accept_completion_slash_command(input_state: InputState) -> None:
    input_state.set_text("/hel")
    input_state.accept_completion("/help")
    assert input_state.text == "/help"


def test_accept_completion_slash_with_suffix(input_state: InputState) -> None:
    input_state.set_text("/hel")
    input_state.accept_completion("/help", with_suffix=True)
    assert input_state.text == "/help "


def test_accept_completion_shell_word(input_state: InputState) -> None:
    input_state.set_text("!git sta")
    input_state.accept_completion("status")
    assert input_state.text == "!git status"


def test_accept_completion_shell_with_suffix(input_state: InputState) -> None:
    input_state.set_text("!git sta")
    input_state.accept_completion("status", with_suffix=True)
    assert input_state.text == "!git status "


# ── History ──────────────────────────────────────────────────────────


def test_history_append_adds_to_list(input_state: InputState) -> None:
    input_state.append_history("first")
    input_state.append_history("second")
    assert input_state.history == ["first", "second"]


def test_history_append_no_disk_write(input_state: InputState) -> None:
    """append_history updates in-memory list only — no disk I/O."""
    input_state.append_history("/review")
    input_state.append_history("!git status")
    assert input_state.history == ["/review", "!git status"]


def test_history_append_bounds_at_max(input_state: InputState) -> None:
    """In-memory history is capped at config.tui.max_history."""
    limit = config.tui.max_history
    input_state.history = [f"cmd{i}" for i in range(limit)]
    input_state.append_history("new")
    assert len(input_state.history) == limit
    assert input_state.history[-1] == "new"
    assert input_state.history[0] == "cmd1"


# ── History prefix search ───────────────────────────────────────────


def test_history_up_cycles_all_when_empty(
    input_state: InputState, input_reader: InputReader
) -> None:
    input_state.history = ["a", "b", "c"]
    input_reader._history_up()
    assert input_state.text == "c"
    input_reader._history_up()
    assert input_state.text == "b"
    input_reader._history_up()
    assert input_state.text == "a"


def test_history_up_filters_by_prefix(input_state: InputState, input_reader: InputReader) -> None:
    input_state.history = ["!git status", "/review", "!git log", "hello"]
    input_state.set_text("!git")
    input_reader._history_up()
    assert input_state.text == "!git log"
    input_reader._history_up()
    assert input_state.text == "!git status"


def test_history_up_stops_at_top(input_state: InputState, input_reader: InputReader) -> None:
    input_state.history = ["!git status", "/review"]
    input_state.set_text("!git")
    input_reader._history_up()
    assert input_state.text == "!git status"
    input_reader._history_up()
    assert input_state.text == "!git status"


def test_history_down_restores_original(input_state: InputState, input_reader: InputReader) -> None:
    input_state.history = ["abc", "abd", "xyz"]
    input_state.set_text("ab")
    input_reader._history_up()
    assert input_state.text == "abd"
    input_reader._history_down()
    assert input_state.text == "ab"


def test_history_down_filters_forward(input_state: InputState, input_reader: InputReader) -> None:
    input_state.history = ["!git status", "/review", "!git log"]
    input_state.set_text("!git")
    input_reader._history_up()  # → !git log
    input_reader._history_up()  # → !git status
    input_reader._history_down()  # → !git log
    assert input_state.text == "!git log"


def test_history_prefix_locked_on_first_up(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Prefix is saved from original text, not from history entries."""
    input_state.history = ["!git status", "!git log --oneline"]
    input_state.set_text("!git l")
    input_reader._history_up()
    assert input_state.text == "!git log --oneline"
    input_reader._history_up()
    assert input_state.text == "!git log --oneline"


# ── History provider (DB-backed) ────────────────────────────────────


def test_provider_seeds_history() -> None:
    """History provider seeds the in-memory list (reversed to chronological)."""
    data = ["newest", "middle", "oldest"]
    state = InputState(history_provider=lambda _prefix, _limit: data)
    reader = InputReader(state)
    reader._load_history()
    assert state.history == ["oldest", "middle", "newest"]


def test_provider_empty_leaves_history_empty() -> None:
    """When provider returns empty, history stays empty."""
    state = InputState(history_provider=lambda _prefix, _limit: [])
    reader = InputReader(state)
    reader._load_history()
    assert state.history == []


def test_no_provider_leaves_history_empty(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Without a provider, history stays empty (no legacy fallback)."""
    input_reader._load_history()
    assert input_state.history == []


# ── Paste markers ────────────────────────────────────────────────────


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


def test_paste_below_threshold_inserts_inline(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Short paste (< 4 lines) inserts text directly, no marker."""
    input_reader._on_key(KeyPress(Keys.BracketedPaste, "line1\nline2"))
    assert input_state.text == "line1\nline2"
    assert input_state.paste_regions == []


def test_paste_above_line_threshold_creates_marker(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Paste with ≥ 4 lines collapses to a marker in the buffer."""
    content = "line1\nline2\nline3\nline4"
    input_reader._on_key(KeyPress(Keys.BracketedPaste, content))
    assert input_state.text == "[pasted 4 lines]"
    assert len(input_state.paste_regions) == 1
    assert input_state.paste_regions[0].content == content


def test_paste_above_char_threshold_creates_marker(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Long single-line paste (> 200 chars) collapses to a char marker."""
    content = "x" * 250
    input_reader._on_key(KeyPress(Keys.BracketedPaste, content))
    assert input_state.text == "[pasted 250 chars]"
    assert input_state.paste_regions[0].content == content


def test_paste_at_cursor_position(input_state: InputState, input_reader: InputReader) -> None:
    """Marker is inserted at cursor position, not appended."""
    input_state.set_text("hello  world", cursor=6)
    input_reader._on_key(KeyPress(Keys.BracketedPaste, "a\nb\nc\nd"))
    assert input_state.text == "hello [pasted 4 lines] world"


# ── Submit expands markers ───────────────────────────────────────────


def test_submit_expands_markers(input_state: InputState, input_reader: InputReader) -> None:
    """Enter submits the real pasted content, not the marker text."""
    content = "line1\nline2\nline3\nline4"
    input_reader._on_key(KeyPress(Keys.BracketedPaste, content))
    input_reader._on_key(KeyPress(Keys.Enter, "\r"))
    submitted = input_state.submitted.get_nowait()
    assert submitted == content


def test_submit_expands_mixed_text_and_markers(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Text typed around a marker is preserved in the submitted string."""
    input_state.set_text("review this: ")
    content = "a\nb\nc\nd"
    input_reader._on_key(KeyPress(Keys.BracketedPaste, content))
    input_reader._on_key(KeyPress(Keys.Enter, "\r"))
    submitted = input_state.submitted.get_nowait()
    assert submitted == f"review this: {content}"


# ── expand_markers / marker_span_at (pure helpers) ───────────────────


def test_expand_markers_no_regions(input_state: InputState) -> None:
    input_state.set_text("no markers here")
    assert input_state.expand_markers(input_state.text) == "no markers here"


def test_expand_markers_multiple_regions(input_state: InputState) -> None:
    input_state.set_text("[pasted 4 lines] and [pasted 200 chars]")
    input_state.paste_regions.append(PasteRegion(marker="[pasted 4 lines]", content="a\nb\nc\nd"))
    input_state.paste_regions.append(PasteRegion(marker="[pasted 200 chars]", content="z" * 200))
    assert input_state.expand_markers(input_state.text) == "a\nb\nc\nd and " + "z" * 200


def test_marker_span_at_inside_returns_span_and_region(input_state: InputState) -> None:
    input_state.set_text("abc[pasted 5 lines]xyz")
    region = PasteRegion(marker="[pasted 5 lines]", content="...")
    input_state.paste_regions.append(region)
    span = input_state.marker_span_at(3)
    assert span is not None
    assert span[:2] == (3, 19)
    assert span[2] is region
    # Last char inside marker (pos 18, exclusive end 19).
    assert input_state.marker_span_at(18) is not None


def test_marker_span_at_outside_returns_none(input_state: InputState) -> None:
    input_state.set_text("abc[pasted 5 lines]xyz")
    input_state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="..."))
    assert input_state.marker_span_at(2) is None
    assert input_state.marker_span_at(19) is None


# ── remove_marker ────────────────────────────────────────────────────


def test_remove_marker_deletes_text_and_region(input_state: InputState) -> None:
    input_state.set_text("abc[pasted 5 lines]xyz")
    region = PasteRegion(marker="[pasted 5 lines]", content="real")
    input_state.paste_regions.append(region)
    input_state.remove_marker(region)
    assert input_state.text == "abcxyz"
    assert input_state.paste_regions == []


def test_remove_marker_cursor_after_marker_shifts_left(input_state: InputState) -> None:
    input_state.set_text("abc[pasted 5 lines]xyz", cursor=19)
    region = PasteRegion(marker="[pasted 5 lines]", content="real")
    input_state.paste_regions.append(region)
    input_state.remove_marker(region)
    assert input_state.cursor == 3


def test_remove_marker_cursor_inside_marker_snaps_to_start(input_state: InputState) -> None:
    """Cursor inside marker (defensive) collapses to marker start."""
    input_state.set_text("abc[pasted 5 lines]xyz", cursor=10)
    region = PasteRegion(marker="[pasted 5 lines]", content="real")
    input_state.paste_regions.append(region)
    input_state.remove_marker(region)
    assert input_state.text == "abcxyz"
    assert input_state.cursor == 3


def test_remove_marker_cursor_before_marker_unchanged(input_state: InputState) -> None:
    input_state.set_text("abc[pasted 5 lines]xyz", cursor=1)
    region = PasteRegion(marker="[pasted 5 lines]", content="real")
    input_state.paste_regions.append(region)
    input_state.remove_marker(region)
    assert input_state.cursor == 1


# ── prune_orphaned_regions ───────────────────────────────────────────


def test_prune_removes_orphaned_regions(input_state: InputState) -> None:
    input_state.set_text("only [pasted 5 lines] remains")
    input_state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="a"))
    input_state.paste_regions.append(PasteRegion(marker="[pasted 3 lines]", content="b"))
    input_state.prune_orphaned_regions()
    assert len(input_state.paste_regions) == 1
    assert input_state.paste_regions[0].marker == "[pasted 5 lines]"


# ── reset clears regions ─────────────────────────────────────────────


def test_reset_clears_paste_regions(input_state: InputState) -> None:
    input_state.set_text("[pasted 5 lines]")
    input_state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="a"))
    input_state.reset()
    assert input_state.text == ""
    assert input_state.paste_regions == []


# ── Atomic cursor navigation via key dispatch ────────────────────────


def test_key_left_snaps_past_marker(input_state: InputState, input_reader: InputReader) -> None:
    """Left arrow one step into a marker snaps cursor to marker start."""
    input_state.set_text("abc[pasted 5 lines]xyz", cursor=19)
    input_state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="..."))
    input_reader._on_key(KeyPress(Keys.Left, ""))
    assert input_state.cursor == 3


def test_key_right_snaps_past_marker(input_state: InputState, input_reader: InputReader) -> None:
    """Right arrow at marker start snaps cursor to marker end."""
    input_state.set_text("abc[pasted 5 lines]xyz", cursor=3)
    input_state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="..."))
    input_reader._on_key(KeyPress(Keys.Right, ""))
    assert input_state.cursor == 19


# ── Atomic deletion via key dispatch ─────────────────────────────────


def test_key_backspace_at_marker_end_removes_marker(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Backspace right after a marker removes the entire marker + region."""
    input_state.set_text("abc[pasted 5 lines]xyz", cursor=19)
    input_state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="real"))
    input_reader._on_key(KeyPress(Keys.Backspace, "\x7f"))
    assert input_state.text == "abcxyz"
    assert input_state.paste_regions == []
    assert input_state.cursor == 3


def test_key_delete_at_marker_start_removes_marker(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Delete key at marker start removes the entire marker + region."""
    input_state.set_text("abc[pasted 5 lines]xyz", cursor=3)
    input_state.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="real"))
    input_reader._on_key(KeyPress(Keys.Delete, ""))
    assert input_state.text == "abcxyz"
    assert input_state.paste_regions == []


def test_key_ctrl_w_removes_overlapping_marker(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Ctrl+W with cursor after marker removes the marker atomically."""
    input_state.set_text("hello [pasted 4 lines]", cursor=22)
    input_state.paste_regions.append(PasteRegion(marker="[pasted 4 lines]", content="a\nb\nc\nd"))
    input_reader._on_key(KeyPress(Keys.ControlW, ""))
    # Ctrl+W kills back to word boundary; the marker overlaps → whole marker gone.
    assert "[pasted" not in input_state.text
    assert input_state.paste_regions == []


def test_key_ctrl_u_removes_overlapping_marker(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Ctrl+U kills to line start, removing any marker in the range."""
    input_state.set_text("abc[pasted 4 lines]xyz", cursor=22)
    input_state.paste_regions.append(PasteRegion(marker="[pasted 4 lines]", content="data"))
    input_reader._on_key(KeyPress(Keys.ControlU, ""))
    assert input_state.paste_regions == []


# ── Multiple pastes ──────────────────────────────────────────────────


def test_multiple_pastes_expand_independently(
    input_state: InputState, input_reader: InputReader
) -> None:
    """Two pastes produce two markers; expand replaces both."""
    input_reader._on_key(KeyPress(Keys.BracketedPaste, "a\nb\nc\nd"))
    input_state.buffer.insert_text(" ")
    input_reader._on_key(KeyPress(Keys.BracketedPaste, "e\nf\ng\nh"))
    assert len(input_state.paste_regions) == 2
    expanded = input_state.expand_markers(input_state.text)
    assert expanded == "a\nb\nc\nd e\nf\ng\nh"


def test_multiple_pastes_delete_one_preserves_other(input_state: InputState) -> None:
    """Deleting one marker preserves the other."""
    input_state.set_text("[pasted 4 lines] [pasted 4 lines #2]")
    r1 = PasteRegion(marker="[pasted 4 lines]", content="first")
    r2 = PasteRegion(marker="[pasted 4 lines #2]", content="second")
    input_state.paste_regions.extend([r1, r2])
    input_state.remove_marker(r1)
    assert "[pasted 4 lines #2]" in input_state.text
    assert len(input_state.paste_regions) == 1
    assert input_state.paste_regions[0].content == "second"


# ── ContextRegion + context_regions ──────────────────────────────────


def test_context_region_fields() -> None:
    r = ContextRegion(marker="[/review]", content="Selected PR.")
    assert r.marker == "[/review]"
    assert r.content == "Selected PR."


def test_context_regions_starts_empty(input_state: InputState) -> None:
    assert input_state.context_regions == []


def test_add_context_appends(input_state: InputState) -> None:
    input_state.add_context("[/review]", "Selected PR.")
    assert len(input_state.context_regions) == 1
    assert input_state.context_regions[0].marker == "[/review]"
    assert input_state.context_regions[0].content == "Selected PR."
    input_state.add_context("[/model]", "Switched.")
    assert len(input_state.context_regions) == 2
    assert input_state.context_regions[1].marker == "[/model]"


def test_pop_context_removes_last(input_state: InputState) -> None:
    input_state.add_context("[/review]", "Selected PR.")
    input_state.add_context("[/model]", "Switched.")
    popped = input_state.pop_context()
    assert popped is not None
    assert popped.marker == "[/model]"
    assert len(input_state.context_regions) == 1
    assert input_state.context_regions[0].marker == "[/review]"


def test_pop_context_empty_returns_none(input_state: InputState) -> None:
    assert input_state.pop_context() is None


def test_clear_context(input_state: InputState) -> None:
    input_state.add_context("[/review]", "Selected PR.")
    input_state.add_context("[/model]", "Switched.")
    input_state.clear_context()
    assert input_state.context_regions == []


def test_reset_clears_context_regions(input_state: InputState) -> None:
    input_state.add_context("[/review]", "Selected PR.")
    input_state.set_text("hello")
    input_state.reset()
    assert input_state.context_regions == []
    assert input_state.text == ""


def test_reset_buffer_preserves_context_regions(input_state: InputState) -> None:
    input_state.add_context("[/review]", "Selected PR.")
    input_state.set_text("/help")
    input_state.paste_regions.append(PasteRegion(marker="[pasted]", content="data"))
    input_state.reset_buffer()
    assert input_state.text == ""
    assert input_state.paste_regions == []
    assert len(input_state.context_regions) == 1
    assert input_state.context_regions[0].marker == "[/review]"
