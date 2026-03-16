"""Shell tool — `run_command` for LLM-callable shell execution.

Delegates to `rbtr.shell_exec` for subprocess mechanics.
This module handles tool-specific concerns: streaming output
via `ToolCallOutput` events and result formatting.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from pydantic_ai import RunContext

from rbtr.config import config
from rbtr.events import ToolCallOutput
from rbtr.llm.deps import AgentDeps
from rbtr.llm.tools.common import shell_toolset
from rbtr.shell_exec import run_shell, truncate_output

# ── Streaming display buffer ─────────────────────────────────────────


@dataclass
class HeadTailBuffer:
    """Rolling head/tail buffer for streaming output.

    Captures the first `head_max` lines (frozen) and the last
    `tail_max` lines (rolling).  Used to build `ToolCallOutput`
    events for the TUI.
    """

    head_max: int = 3
    tail_max: int = 5
    head: list[str] = field(default_factory=list)
    tail: deque[str] = field(default_factory=deque)
    total_lines: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tail = deque(maxlen=self.tail_max)

    def add_line(self, line: str) -> None:
        """Append a line of output."""
        self.total_lines += 1
        if len(self.head) < self.head_max:
            self.head.append(line)
        self.tail.append(line)

    @property
    def elapsed(self) -> float:
        """Seconds since the buffer was created."""
        return time.monotonic() - self.started_at

    def to_event(self, tool_name: str) -> ToolCallOutput:
        """Build a `ToolCallOutput` event from the current state."""
        return ToolCallOutput(
            tool_name=tool_name,
            head="\n".join(self.head),
            tail="\n".join(self.tail),
            total_lines=self.total_lines,
            elapsed=self.elapsed,
        )


# ── Tool ─────────────────────────────────────────────────────────────


@shell_toolset.tool
def run_command(
    ctx: RunContext[AgentDeps],
    command: str,
    timeout: int | None = None,
) -> str:
    """Run a shell command and return its output.

    Primary use: executing scripts bundled with skills.

    **Do not use for codebase access.** The working directory
    may not match the review target — files under review
    live in a different branch or commit. Use `read_file`,
    `grep`, `list_files`, `search`, `read_symbol`, and other
    bespoke tools instead — they read from the git object
    store at the correct ref, are paginated, and are faster.

    The repository should not be modified by commands run
    here. Treat the working tree as read-only.

    Args:
        command: Shell command to execute.
        timeout: Max seconds (default from config, 0 = no limit).
    """
    shell_cfg = config.tools.shell
    effective_timeout = timeout if timeout is not None else shell_cfg.timeout
    events_queue = ctx.deps.events

    buf = HeadTailBuffer()
    last_emit = 0.0

    def _on_line(line: str) -> None:
        nonlocal last_emit
        buf.add_line(line)
        now = time.monotonic()
        if (now - last_emit) >= 0.033:  # ~30 fps
            events_queue.put(buf.to_event("run_command"))
            last_emit = now

    result = run_shell(
        command,
        cancel=ctx.deps.cancel,
        timeout=float(effective_timeout),
        on_line=_on_line,
    )

    # Final event so the TUI shows the complete state.
    events_queue.put(buf.to_event("run_command"))

    # Format the result for the model.
    parts: list[str] = []
    if result.stdout:
        stdout, hidden = truncate_output(result.stdout, shell_cfg.max_output_lines)
        parts.append(stdout)
        if hidden:
            parts.append(f"\n… {hidden} lines truncated")
    if result.stderr:
        parts.append(f"(stderr)\n{result.stderr}")
    if result.returncode != 0:
        parts.append(f"exit code {result.returncode}")

    return "\n".join(parts) if parts else "(no output)"
