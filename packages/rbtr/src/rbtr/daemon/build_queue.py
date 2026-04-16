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
    """Serialised build queue with a single worker thread."""

    def __init__(self, mgr: RepoManager, notify: NotifyFn) -> None:
        self._mgr = mgr
        self._notify = notify
        self._queue: queue.SimpleQueue[tuple[str, list[str]]] = queue.SimpleQueue()
        self._shutdown = False
        self.active_repo: str | None = None

    def submit(self, repo: str, refs: list[str]) -> None:
        """Enqueue a build."""
        self._queue.put((repo, refs))

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
            self.active_repo = repo
            try:
                self._build(repo, refs)
            except Exception:
                log.exception("Build failed for %s", repo)
            finally:
                self.active_repo = None

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
