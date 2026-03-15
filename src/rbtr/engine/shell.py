"""Handler for `!commands` — user-facing shell execution.

Delegates to `rbtr.shell_exec` for the subprocess mechanics.
This module handles engine-specific concerns: event emission,
context markers, and expandable output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.config import config
from rbtr.events import OutputLevel
from rbtr.shell_exec import ShellResult, run_shell, truncate_output

if TYPE_CHECKING:
    from .core import Engine


def handle_shell(engine: Engine, cmd: str) -> None:
    """Run a shell command, streaming truncated output as events."""

    if not cmd:
        engine._out("Usage: !<command>")
        return
    engine._out(f"$ {cmd}")

    result = run_shell(
        cmd,
        cancel=engine._cancel,
    )

    _render_result(engine, cmd, result)


def _render_result(engine: Engine, cmd: str, result: ShellResult) -> None:
    """Emit output events and context marker for a completed shell command."""
    total_hidden = 0
    if result.stdout:
        shown, hidden = truncate_output(result.stdout, config.tui.shell_max_lines)
        engine._out(shown)
        total_hidden += hidden
    if result.stderr:
        shown, hidden = truncate_output(result.stderr, config.tui.shell_max_lines)
        engine._out(shown, level=OutputLevel.SHELL_STDERR)
        total_hidden += hidden

    had_error = result.returncode != 0
    if had_error and not engine._cancel.is_set():
        engine._error(f"(exit code {result.returncode})")

    if total_hidden:
        engine._last_shell_full_output = (
            result.stdout,
            result.stderr,
            result.returncode,
            total_hidden,
        )
    else:
        engine._last_shell_full_output = None

    _emit_shell_context(engine, cmd, result)


def _emit_shell_context(engine: Engine, cmd: str, result: ShellResult) -> None:
    """Emit a `ContextMarkerReady` event summarising the shell command."""
    marker = f"[! {cmd} — exit {result.returncode}]"

    max_chars = config.tui.shell_context_max_chars
    parts: list[str] = [f"$ {cmd}"]
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"(stderr)\n{result.stderr}")
    parts.append(f"exit code {result.returncode}")
    body = "\n".join(parts)

    if len(body) > max_chars:
        body = body[:max_chars] + "\n… (truncated)"

    engine._context(marker, body)
