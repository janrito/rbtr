"""Progress reporting for index build, embed, and collection.

Two callback shapes, one per thing being worked through: `build_index`
and `embed_index` move through phases of one repo, while `run_gc_all`
moves through the repos themselves.
"""

from __future__ import annotations

from collections.abc import Callable

type ProgressCallback = Callable[[str, int, int], None]
"""`(phase, done, total)` — called to report build/embed progress."""

type RepoProgressCallback = Callable[[str, int, int], None]
"""`(repo_path, done, total)` — called per repo of a global collection."""


def _noop_progress(_phase: str, _done: int, _total: int) -> None:
    pass


def _noop_repo_progress(_repo_path: str, _done: int, _total: int) -> None:
    pass
