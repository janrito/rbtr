"""Handler for !commands — shell execution with pty and truncation."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from typing import TYPE_CHECKING

from rbtr.config import config
from rbtr.events import OutputLevel
from rbtr.exceptions import TaskCancelled

if TYPE_CHECKING:
    from .core import Engine


def handle_shell(engine: Engine, cmd: str) -> None:
    """Run a shell command, streaming truncated output as events."""

    if not cmd:
        engine._out("Usage: !<command>")
        return
    engine._out(f"$ {cmd}")
    # Give the child a pty as stdin so terminal ioctls work
    # (e.g. watch, less), but our reader thread keeps exclusive
    # access to the real stdin.
    master_fd, slave_fd = os.openpty()
    try:
        proc = subprocess.Popen(  # noqa: S602
            cmd,
            shell=True,
            stdin=slave_fd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    finally:
        os.close(slave_fd)
    engine._shell_proc = proc
    try:
        engine._shell_pgid = os.getpgid(proc.pid)
    except OSError:
        engine._shell_pgid = proc.pid
    try:
        # communicate(timeout) drains pipes properly (unlike wait).
        # First call starts internal reader threads; subsequent calls
        # just re-join them — safe to call in a loop.
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.02)
                break
            except subprocess.TimeoutExpired:
                engine._check_cancel()
    except TaskCancelled:
        # SIGKILL the entire process group — SIGTERM from cancel()
        # may have been trapped by the child (e.g. watch, less).
        with contextlib.suppress(OSError):
            os.killpg(engine._shell_pgid, signal.SIGKILL)
        with contextlib.suppress(OSError):
            proc.wait(timeout=2)
        raise
    finally:
        engine._shell_proc = None
        engine._shell_pgid = None
        os.close(master_fd)
    engine._check_cancel()
    stdout_full = stdout.rstrip() if stdout else ""
    stderr_full = stderr.rstrip() if stderr else ""
    total_hidden = 0
    if stdout_full:
        shown, hidden = _truncate_output(stdout_full, config.tui.shell_max_lines)
        engine._out(shown)
        total_hidden += hidden
    if stderr_full:
        shown, hidden = _truncate_output(stderr_full, config.tui.shell_max_lines)
        engine._out(shown, level=OutputLevel.SHELL_STDERR)
        total_hidden += hidden
    had_error = proc.returncode != 0
    if had_error and not engine._cancel.is_set():
        engine._error(f"(exit code {proc.returncode})")
    if total_hidden:
        engine._last_shell_full_output = (stdout_full, stderr_full, proc.returncode, total_hidden)
    else:
        engine._last_shell_full_output = None

    # Emit context marker so the LLM knows what was run.
    _emit_shell_context(engine, cmd, stdout_full, stderr_full, proc.returncode)


def _emit_shell_context(
    engine: Engine,
    cmd: str,
    stdout: str,
    stderr: str,
    returncode: int,
) -> None:
    """Emit a `ContextMarkerReady` event summarising the shell command."""
    rc = returncode
    marker = f"[! {cmd} — exit {rc}]"

    max_chars = config.tui.shell_context_max_chars
    parts: list[str] = [f"$ {cmd}"]
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"(stderr)\n{stderr}")
    parts.append(f"exit code {rc}")
    body = "\n".join(parts)

    if len(body) > max_chars:
        body = body[:max_chars] + "\n… (truncated)"

    engine._context(marker, body)


def _truncate_output(text: str, max_lines: int) -> tuple[str, int]:
    """Truncate text to max_lines. Returns (truncated, hidden_count)."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text, 0
    truncated = "\n".join(lines[:max_lines])
    return truncated, len(lines) - max_lines
