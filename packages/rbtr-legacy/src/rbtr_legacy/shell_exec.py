"""Shared shell execution core.

One implementation, two entry points: the user-facing `!cmd`
(`engine/shell.py`) and the LLM-callable `run_command` tool
(`llm/tools/shell.py`).  Both delegate here.

The core runs a subprocess with PTY allocation, process-group
management, output collection, timeout enforcement, and clean
cancellation.  A per-line callback lets callers observe output
as it arrives without coupling the core to any display strategy.
"""

from __future__ import annotations

import contextlib
import os
import select
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO

from rbtr_legacy.exceptions import TaskCancelled


@dataclass
class ShellResult:
    """Result of a shell command execution."""

    stdout: str
    stderr: str
    returncode: int


type LineCallback = Callable[[str], None]


def run_shell(
    cmd: str,
    *,
    cancel: threading.Event,
    timeout: float = 0,
    on_line: LineCallback | None = None,
) -> ShellResult:
    """Run a shell command with PTY, cancel support, and per-line streaming.

    Args:
        cmd: Shell command to execute.
        cancel: Set this event to request cancellation.
        timeout: Max seconds (0 = no limit).
        on_line: Called for each line of stdout read during
            execution.  Must not block.

    Returns:
        `ShellResult` with stdout, stderr, and return code.

    Raises:
        TaskCancelled: If *cancel* was set during execution.
    """
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
    except Exception:
        os.close(master_fd)
        raise
    finally:
        os.close(slave_fd)

    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid

    deadline = (time.monotonic() + timeout) if timeout > 0 else 0.0

    def _kill() -> None:
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGKILL)
        with contextlib.suppress(OSError):
            proc.wait(timeout=2)

    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                if cancel.is_set():
                    _kill()
                    raise TaskCancelled from None
                if deadline and time.monotonic() > deadline:
                    _kill()
                    return ShellResult(
                        stdout="",
                        stderr=f"Command timed out after {timeout}s",
                        returncode=-1,
                    )
                # Drain available stdout for streaming.
                if on_line:
                    _drain_pipe(proc.stdout, on_line)
    except TaskCancelled:
        raise
    except Exception:
        _kill()
        raise
    finally:
        os.close(master_fd)

    if cancel.is_set():
        raise TaskCancelled

    # Deliver any remaining lines from the final output.
    if on_line:
        for line in (stdout or "").rstrip().splitlines():
            on_line(line)

    return ShellResult(
        stdout=(stdout or "").rstrip(),
        stderr=(stderr or "").rstrip(),
        returncode=proc.returncode or 0,
    )


def truncate_output(text: str, max_lines: int) -> tuple[str, int]:
    """Truncate *text* to *max_lines*.  Returns `(truncated, hidden_count)`."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text, 0
    truncated = "\n".join(lines[:max_lines])
    return truncated, len(lines) - max_lines


def truncate_for_agent(
    text: str,
    max_lines: int = 2000,
    max_bytes: int = 51_200,
) -> str:
    """Truncate *text* to *max_lines* or *max_bytes* (whichever hits first).

    When truncation is needed, keeps equal head and tail portions
    with a divider indicating how many lines were dropped.
    """
    if not text:
        return text

    lines = text.splitlines()

    if len(lines) <= max_lines and len(text.encode()) <= max_bytes:
        return text

    # Start with half-and-half line budget.
    half = max_lines // 2
    head_n = half
    tail_n = max_lines - head_n

    # Clamp to actual line count.
    if head_n + tail_n >= len(lines):
        head_n = len(lines) // 2
        tail_n = len(lines) - head_n

    head = lines[:head_n]
    tail = lines[len(lines) - tail_n :] if tail_n else []
    hidden = len(lines) - head_n - tail_n

    # Shrink head until the assembled result fits the byte budget.
    while head_n > 0:
        divider = f"[… {hidden} lines truncated …]"
        parts = [*head, divider, *tail]
        assembled = "\n".join(parts)
        if len(assembled.encode()) <= max_bytes:
            return assembled
        # Drop one line from the end of head.
        head_n -= 1
        head = lines[:head_n]
        hidden = len(lines) - head_n - tail_n

    # Head is empty — try shrinking tail.
    while tail_n > 0:
        divider = f"[… {hidden} lines truncated …]"
        parts = [divider, *tail]
        assembled = "\n".join(parts)
        if len(assembled.encode()) <= max_bytes:
            return assembled
        tail_n -= 1
        tail = lines[len(lines) - tail_n :] if tail_n else []
        hidden = len(lines) - tail_n

    # Everything truncated.
    return f"[… {len(lines)} lines truncated …]"


def _drain_pipe(pipe: IO[str] | None, callback: LineCallback) -> None:
    """Read available lines from a pipe without blocking."""
    if pipe is None:
        return
    while True:
        ready, _, _ = select.select([pipe], [], [], 0)
        if not ready:
            break
        line = pipe.readline()
        if not line:
            break
        callback(line.rstrip("\n"))
