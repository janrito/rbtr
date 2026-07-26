"""Progress reporting for index build and embed.

A single callback shape shared by `build_index` and `embed_index` and the
daemon job worker that drives them.
"""

from __future__ import annotations

from collections.abc import Callable

type ProgressCallback = Callable[[str, int, int], None]
"""`(phase, done, total)` — called to report build/embed progress."""


def _noop_progress(_phase: str, _done: int, _total: int) -> None:
    pass
