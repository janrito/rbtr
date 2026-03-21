"""Shared test helpers — event queue utilities."""

from __future__ import annotations

import queue

from rbtr.events import Event, MarkdownOutput, Output


def drain(events: queue.Queue[Event]) -> list[Event]:
    """Drain all events from the queue into a list."""
    result: list[Event] = []
    while True:
        try:
            result.append(events.get_nowait())
        except queue.Empty:
            break
    return result


def output_texts(events: list[Event]) -> list[str]:
    """Extract text from Output and MarkdownOutput events."""
    texts: list[str] = []
    for e in events:
        if isinstance(e, (Output, MarkdownOutput)):
            texts.append(e.text)
    return texts


def has_event_type(events: list[Event], event_type: type) -> bool:
    """Check whether any event matches the given type."""
    return any(isinstance(e, event_type) for e in events)
