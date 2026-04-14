"""Tests for Live viewport behaviour in the Rich TUI."""

from __future__ import annotations

import queue

from pytest_mock import MockerFixture
from rich.console import Console, RenderableType
from rich.text import Text

from rbtr_legacy.state import EngineState
from rbtr_legacy.styles import THEME
from rbtr_legacy.tui.input import PasteRegion
from rbtr_legacy.tui.ui import UI, _LivePanel, _render_lines, _tail_renderable_lines


def _plain_lines(console: Console, renderable: RenderableType) -> list[str]:
    """Render *renderable* and return plain-text lines."""
    lines = _render_lines(console, renderable)
    return ["".join(seg.text for seg in line if not seg.control) for line in lines]


def test_tail_renderable_lines_keeps_bottom_lines() -> None:
    console = Console(width=40, height=10, force_terminal=True, theme=THEME)
    content = Text("\n".join(f"line {n}" for n in range(1, 8)))

    clipped = _tail_renderable_lines(console, content, 3)

    assert _plain_lines(console, clipped) == ["line 5", "line 6", "line 7"]


def test_render_view_keeps_input_visible_with_tall_active_panel(mocker: MockerFixture) -> None:
    console = Console(width=80, height=12, force_terminal=True, theme=THEME)
    state = EngineState()
    engine = mocker.MagicMock()
    engine._last_shell_full_output = None
    ui = UI(console, state, queue.Queue(), engine)

    ui._active_task = True
    ui._active_lines = [Text("\n".join(f"row {n}" for n in range(1, 60)))]

    lines = _plain_lines(console, ui._render_view())

    assert any(line.startswith("> ") for line in lines)
    assert any("row 59" in line for line in lines)


def test_render_view_keeps_input_visible_with_tall_pending_panel(mocker: MockerFixture) -> None:
    console = Console(width=80, height=12, force_terminal=True, theme=THEME)
    state = EngineState()
    engine = mocker.MagicMock()
    engine._last_shell_full_output = None
    ui = UI(console, state, queue.Queue(), engine)

    ui._live_panels["_task"] = _LivePanel(
        lines=[Text("\n".join(f"pending {n}" for n in range(1, 60)))],
        variant="succeeded",
        done=True,
    )

    lines = _plain_lines(console, ui._render_view())

    assert any(line.startswith("> ") for line in lines)
    assert any("pending 59" in line for line in lines)


# ── Paste marker rendering ───────────────────────────────────────────


def _make_ui(mocker: MockerFixture) -> UI:
    """Build a UI wired to a mock engine for rendering tests."""
    console = Console(width=80, height=10, force_terminal=True, theme=THEME)
    state = EngineState()
    engine = mocker.MagicMock()
    engine._last_shell_full_output = None
    return UI(console, state, queue.Queue(), engine)


def test_render_input_line_styles_paste_marker(mocker: MockerFixture) -> None:
    """Paste markers are rendered with the PASTE_MARKER style."""
    ui = _make_ui(mocker)

    ui.inp.set_text("hello [pasted 5 lines] world", cursor=0)
    ui.inp.paste_regions.append(PasteRegion(marker="[pasted 5 lines]", content="..."))

    rendered = ui._render_input_line()

    assert "[pasted 5 lines]" in rendered.plain
    # Walk rendered segments — the marker must have the italic style
    # from PASTE_MARKER, while surrounding text must not.
    segments = list(rendered.render(ui.console))
    marker_segments = [s for s in segments if "[pasted" in s.text or "lines]" in s.text]
    assert marker_segments, "Marker text should appear in rendered segments"
    assert all("italic" in str(s.style) for s in marker_segments)


def test_render_cursor_on_paste_marker(mocker: MockerFixture) -> None:
    """When cursor is at a marker, the leading '[' gets cursor style."""
    ui = _make_ui(mocker)
    marker = "[pasted 5 lines]"
    # Place cursor exactly at the marker start.
    ui.inp.set_text(f"hello {marker} world", cursor=6)
    ui.inp.paste_regions.append(PasteRegion(marker=marker, content="..."))

    rendered = ui._render_input_line()
    segments = list(rendered.render(ui.console))

    # Find the segment containing the leading '['.
    bracket_seg = [s for s in segments if s.text == "["]
    assert bracket_seg, "Leading '[' should be its own segment"
    assert "reverse" in str(bracket_seg[0].style), "Leading '[' should have cursor style"

    # The rest of the marker should have italic (paste marker) style.
    rest_segs = [s for s in segments if "pasted 5 lines]" in s.text]
    assert rest_segs
    assert all("italic" in str(s.style) for s in rest_segs)


# ── Context marker rendering ────────────────────────────────────────


def test_render_context_line_empty(mocker: MockerFixture) -> None:
    """No context regions → no context line."""
    ui = _make_ui(mocker)
    assert ui._render_context_line() is None


def test_render_context_line_single(mocker: MockerFixture) -> None:
    """Single context region renders its marker text."""
    ui = _make_ui(mocker)
    ui.inp.add_context("[/review → PR #42]", "Selected PR.")
    line = ui._render_context_line()
    assert line is not None
    assert "[/review → PR #42]" in line.plain


def test_render_context_line_multiple(mocker: MockerFixture) -> None:
    """Multiple context regions render space-separated."""
    ui = _make_ui(mocker)
    ui.inp.add_context("[/review → PR #42]", "Selected PR.")
    ui.inp.add_context("[/model → gpt-4o]", "Switched.")
    line = ui._render_context_line()
    assert line is not None
    plain = line.plain
    assert "[/review → PR #42]" in plain
    assert "[/model → gpt-4o]" in plain
    # Order preserved.
    assert plain.index("[/review") < plain.index("[/model")
