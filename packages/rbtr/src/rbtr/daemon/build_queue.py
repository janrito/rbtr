"""Serialised build queue — one build at a time.

Currently one worker because the local GGUF embedding model
is GPU-bound. When API-based embedding providers are added,
scale to N workers by starting more threads from `start()`.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from rbtr.daemon.messages import (
    IndexErrorNotification,
    Notification,
    ProgressNotification,
    ReadyNotification,
)
from rbtr.daemon.repos import RepoManager
from rbtr.git import open_repo
from rbtr.index.orchestrator import build_index

from .handlers import resolve_refs

log = logging.getLogger(__name__)

type NotifyFn = Callable[[Notification], None]


class BuildQueue:
    """Serialised build queue with a single worker thread.

    Two internal structures, two jobs:

    - ``_queue`` blocks the worker thread until work arrives
      (`SimpleQueue.get` with timeout).
    - ``_submitted`` deduplicates — O(1) check whether a repo
      is already queued or actively building. Cleared when the
      build finishes (success or failure).
    """

    def __init__(self, mgr: RepoManager, notify: NotifyFn) -> None:
        self._mgr = mgr
        self._notify = notify
        self._queue: queue.SimpleQueue[tuple[str, list[str]]] = queue.SimpleQueue()
        self._submitted: set[str] = set()
        self._shutdown = False

    def submit(self, repo: str, refs: list[str]) -> bool:
        """Enqueue a build. Returns False if already submitted."""
        if repo in self._submitted:
            return False
        self._submitted.add(repo)
        self._queue.put((repo, refs))
        return True

    def is_busy(self, repo: str) -> bool:
        """Check if a repo has a queued or active build."""
        return repo in self._submitted

    def start(self) -> threading.Thread:
        """Start the worker thread."""
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._shutdown = True

    def _worker(self) -> None:
        while not self._shutdown:
            try:
                repo, refs = self._queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._build(repo, refs)
            except Exception:
                log.exception("Build failed for %s", repo)
            finally:
                self._submitted.discard(repo)

    def _build(self, repo_path: str, refs: list[str]) -> None:
        repo = open_repo(repo_path)
        repo_id = self._mgr.resolve(repo_path)

        resolved = resolve_refs(repo, refs)
        if not isinstance(resolved, list):
            self._notify(
                IndexErrorNotification(
                    repo=repo_path,
                    message=resolved.message,
                )
            )
            return

        for sha in resolved:

            def on_progress(done: int, total: int) -> None:
                self._notify(
                    ProgressNotification(
                        repo=repo_path,
                        phase="parsing",
                        current=done,
                        total=total,
                    )
                )

            def on_embed_progress(done: int, total: int) -> None:
                self._notify(
                    ProgressNotification(
                        repo=repo_path,
                        phase="embedding",
                        current=done,
                        total=total,
                    )
                )

            result = build_index(
                repo,
                sha,
                self._mgr.store,
                repo_id=repo_id,
                on_progress=on_progress,
                on_embed_progress=on_embed_progress,
            )

            self._notify(
                ReadyNotification(
                    repo=repo_path,
                    ref=sha,
                    chunks=result.stats.total_chunks,
                    edges=result.stats.total_edges,
                    elapsed=round(result.stats.elapsed_seconds, 2),
                )
            )
