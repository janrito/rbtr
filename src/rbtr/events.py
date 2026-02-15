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


# Union of all event types the UI needs to handle.
Event = (
    TaskStarted | Output | TableOutput | MarkdownOutput | LinkOutput | FlushPanel | TaskFinished
)
