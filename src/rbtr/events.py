"""Typed events for communication between the execution engine and the UI.

The engine produces events. The UI consumes them. Neither imports the other.
All shared state flows through these models.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class OutputLevel(StrEnum):
    """Semantic level for `Output` events.

    The engine sets the level; the TUI maps it to a theme key.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SHELL_STDERR = "shell_stderr"


class TaskStarted(BaseModel):
    """A new task has begun execution."""

    task_type: str


class Output(BaseModel):
    """A line of output from a running task."""

    text: str
    level: OutputLevel = OutputLevel.INFO
    detail: str | None = None


class TableOutput(BaseModel):
    """A table to display. Columns and rows are plain strings."""

    title: str = ""
    columns: list[ColumnDef]
    rows: list[list[str]]


class ColumnDef(BaseModel):
    """Definition of a single table column."""

    header: str
    width: int | None = None
    highlight: bool = False


class MarkdownOutput(BaseModel):
    """Markdown content to render."""

    text: str


class TextDelta(BaseModel):
    """A streaming text chunk from an LLM response.

    The UI accumulates deltas and renders the growing markdown.
    """

    delta: str


class LinkOutput(BaseModel):
    """A link to display. The TUI builds the Rich markup."""

    url: str
    label: str = ""


class FlushPanel(BaseModel):
    """Flush current active lines as a completed panel and start fresh.

    discard=False: print active lines to scrollback as a history panel.
    discard=True: throw away active lines (transient content like
    "Fetching…" that is no longer useful once real data arrives).
    """

    discard: bool = False


class TaskFinished(BaseModel):
    """A task has completed."""

    success: bool
    cancelled: bool = False


# ── Tool call events ──────────────────────────────────────────────────


class ToolCallStarted(BaseModel):
    """The LLM is calling a tool."""

    tool_name: str
    args: str
    tool_call_id: str


class ToolCallOutput(BaseModel):
    """Incremental output from a running tool call.

    Emitted periodically while a long-running tool (e.g.
    `run_command`) executes.  Carries enough state for the
    TUI to render a progress view without buffering itself.
    """

    tool_name: str
    tool_call_id: str
    head: str
    """First few lines (frozen once captured)."""
    tail: str
    """Last few lines (rolling window)."""
    total_lines: int
    elapsed: float
    """Seconds since the command started."""

    @property
    def head_lines(self) -> int:
        """Number of lines in `head`."""
        return self.head.count("\n") + 1 if self.head else 0

    @property
    def tail_lines(self) -> int:
        """Number of lines in `tail`."""
        return self.tail.count("\n") + 1 if self.tail else 0


class ToolCallFinished(BaseModel):
    """A tool call has completed.

    `result` contains the full tool output (up to a generous char
    limit).  The UI is responsible for line-based truncation when
    rendering.

    When `error` is set the tool call failed — `result` is empty
    and `error` contains the error message.
    """

    tool_name: str
    tool_call_id: str
    result: str
    error: str | None = None


# ── Index events ─────────────────────────────────────────────────────


class IndexStarted(BaseModel):
    """Background indexing has begun for a review target."""

    total_files: int


class IndexProgress(BaseModel):
    """Incremental progress update from the indexer.

    `phase` describes what the indexer is doing (e.g. "parsing",
    "embedding").  `indexed` / `total` track the current phase.
    """

    phase: str
    indexed: int
    total: int


class IndexReady(BaseModel):
    """Indexing is complete and the store is queryable.

    `chunk_count` is the total number of chunks in the store.
    """

    chunk_count: int


class IndexCleared(BaseModel):
    """The index has been cleared — no data available."""


# ── Compaction events ────────────────────────────────────────────────


class CompactionStarted(BaseModel):
    """Auto or manual compaction has begun."""

    old_messages: int
    kept_messages: int


class CompactionFinished(BaseModel):
    """Compaction complete — history was replaced."""

    summary_tokens: int


# ── Memory events ────────────────────────────────────────────────────


class FactExtractionStarted(BaseModel):
    """Fact extraction has begun."""


class FactExtractionFinished(BaseModel):
    """Fact extraction complete."""

    added: int = 0
    confirmed: int = 0
    superseded: int = 0


# ── Context marker events ─────────────────────────────────────────────


class ContextMarkerReady(BaseModel):
    """A command produced context for the LLM.

    The TUI appends a `ContextRegion` to
    `InputState.context_regions` (rendered above the prompt,
    not in the editing buffer).  On submit, it expands to
    *content*.  The user can dismiss markers with Backspace
    at cursor position 0.
    """

    marker: str
    content: str


# ── Review events ────────────────────────────────────────────────────


class ReviewPosted(BaseModel):
    """A review was posted to GitHub."""

    url: str


# Union of all event types the UI needs to handle.
Event = (
    TaskStarted
    | Output
    | TableOutput
    | MarkdownOutput
    | TextDelta
    | LinkOutput
    | FlushPanel
    | TaskFinished
    | ToolCallStarted
    | ToolCallOutput
    | ToolCallFinished
    | IndexStarted
    | IndexProgress
    | IndexReady
    | IndexCleared
    | CompactionStarted
    | CompactionFinished
    | ContextMarkerReady
    | FactExtractionStarted
    | FactExtractionFinished
    | ReviewPosted
)
