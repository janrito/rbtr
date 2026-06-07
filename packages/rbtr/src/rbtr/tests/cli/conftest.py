"""Shared fixtures for CLI tests."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    """Capture `emit()`'s rendered (TTY) output into a buffer.

    Redirects the module Console to a forced-terminal, fixed-width
    Console writing to a `StringIO`.  `force_terminal=True` makes
    `_json_output` select the rich path; the fixed `width` keeps
    wrapping deterministic for substring assertions.
    """
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=100, highlight=False)
    monkeypatch.setattr("rbtr.cli.output._out", console)
    return buf
