"""Typed events for communication between the execution engine and the UI.

The engine produces events. The UI consumes them. Neither imports the other.
All shared state flows through these models.
"""

from __future__ import annotations

from pydantic import BaseModel


class TaskStarted(BaseModel):
    """A new task has begun execution."""

    task_id: str


class Output(BaseModel):
    """A line of output from a running task."""

    text: str
    style: str = "dim"


class TableOutput(BaseModel):
    """A table to display. Columns and rows are plain strings."""

    title: str = ""
    columns: list[ColumnDef]
    rows: list[list[str]]
    style: str = "dim"


class ColumnDef(BaseModel):
    """Definition of a single table column."""

    header: str
    width: int | None = None
    style: str = ""


class MarkdownOutput(BaseModel):
    """Markdown content to render."""

    text: str


class TextDelta(BaseModel):
    """A streaming text chunk from an LLM response.

    The UI accumulates deltas and renders the growing markdown.
    """

    delta: str


class LinkOutput(BaseModel):
    """A message containing a Rich markup link."""

    markup: str


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


class ToolCallFinished(BaseModel):
    """A tool call has completed.

    ``result`` contains the full tool output (up to a generous char
    limit).  The UI is responsible for line-based truncation when
    rendering.
    """

    tool_name: str
    result: str


# ── Index events ─────────────────────────────────────────────────────


class IndexStarted(BaseModel):
    """Background indexing has begun for a review target."""

    total_files: int


class IndexProgress(BaseModel):
    """Incremental progress update from the indexer.

    ``phase`` describes what the indexer is doing (e.g. "parsing",
    "embedding").  ``indexed`` / ``total`` track the current phase.
    """

    phase: str
    indexed: int
    total: int


class IndexReady(BaseModel):
    """Indexing is complete and the store is queryable.

    ``chunk_count`` is the total number of chunks in the store.
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
    | ToolCallFinished
    | IndexStarted
    | IndexProgress
    | IndexReady
    | IndexCleared
    | CompactionStarted
    | CompactionFinished
)
