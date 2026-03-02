"""Tests for Live viewport behaviour in the Rich TUI."""

from __future__ import annotations

import queue

from pytest_mock import MockerFixture
from rich.console import Console, RenderableType
from rich.text import Text

from rbtr.state import EngineState
from rbtr.styles import THEME
from rbtr.tui import UI, _render_lines, _tail_renderable_lines


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

    ui._pending_lines = [Text("\n".join(f"pending {n}" for n in range(1, 60)))]
    ui._pending_variant = "succeeded"

    lines = _plain_lines(console, ui._render_view())

    assert any(line.startswith("> ") for line in lines)
    assert any("pending 59" in line for line in lines)
