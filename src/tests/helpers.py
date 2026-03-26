"""Shared test helpers — event queue utilities and lightweight test doubles."""

from __future__ import annotations

import queue
from dataclasses import dataclass

from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings

from rbtr.config import ThinkingEffort
from rbtr.events import Event, MarkdownOutput, Output
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState
from rbtr.tui.input import InputState
from rbtr.tui.ui import UI

# ── Test provider ────────────────────────────────────────────────────


class StubProvider:
    """Provider that returns ``TestModel()`` — no credentials needed.

    Registered in the global conftest under the ``"test"`` prefix so
    that ``build_model("test/...")`` works without mocking.

    Use ``set_model(model)`` to override the returned model for the
    current test.  Resets to ``TestModel()`` after each test.
    """

    GENAI_ID = "test"
    LABEL = "Test"

    def __init__(self) -> None:
        self._model: Model = TestModel()

    def set_model(self, model: Model) -> None:
        """Set the model returned by ``build_model``."""
        self._model = model

    def reset(self) -> None:
        """Reset to default ``TestModel()``.  Called by conftest after each test."""
        self._model = TestModel()

    def is_connected(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return []

    def build_model(self, model_name: str) -> Model:
        return self._model

    def model_settings(
        self, model_id: str, model: Model, effort: ThinkingEffort
    ) -> ModelSettings | None:
        return None

    def context_window(self, model_id: str) -> int | None:
        return 200_000


@dataclass
class MemCtx:
    """Concrete `MemoryContext` for tests — satisfies the protocol structurally."""

    store: SessionStore
    state: EngineState


# ── Headless test doubles ────────────────────────────────────────────


class HeadlessUI(UI):
    """UI without console, Live, or event queue — for completion tests.

    Only `self.inp` is set.  Sufficient for `_complete_shell`
    which only reads `self.inp`.
    """

    def __init__(self, inp: InputState) -> None:
        self.inp = inp


# ── Event queue helpers ──────────────────────────────────────────────


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
