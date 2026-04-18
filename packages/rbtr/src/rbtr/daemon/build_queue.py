"""Serialised build queue — one build at a time.

Currently one worker because the local GGUF embedding model
is GPU-bound. When API-based embedding providers are added,
scale to N workers by starting more threads from `start()`.

Submissions are deduped in-place against the current queue
and the active job. The dedupe key is ``(repo, tuple(refs))`` —
different ref lists for the same repo are distinct intents
and all survive. Callers are expected to pass *resolved* SHAs
so that watcher- and CLI-originated submissions for the same
commit collapse; symbolic-ref dedupe would be a silent miss.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
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
type _QueueKey = tuple[str, tuple[str, ...]]


class BuildQueue:
    """Serialised build queue with a single worker thread.

    The queue is a ``deque`` guarded by a ``threading.Condition``.
    The condition's internal lock is the single serialisation
    point for: queue mutation, dedupe walks, active-job
    tracking, and shutdown signalling.
    """

    def __init__(self, mgr: RepoManager, notify: NotifyFn) -> None:
        self._mgr = mgr
        self._notify = notify
        self._cond = threading.Condition()
        self._queue: deque[_QueueKey] = deque()
        self._shutdown = False
        self.active_repo: str | None = None
        self._active_ref_key: _QueueKey | None = None

    def submit(self, repo: str, refs: list[str]) -> None:
        """Enqueue a build, deduped against the queue and active job.

        No-op (logged at DEBUG) if an identical ``(repo, refs)``
        entry is already queued or currently building.
        """
        key: _QueueKey = (repo, tuple(refs))
        with self._cond:
            if self._active_ref_key == key or key in self._queue:
                log.debug("Dedupe: skipping duplicate submit for %s", key)
                return
            self._queue.append(key)
            self._cond.notify()

    def start(self) -> threading.Thread:
        """Start the worker thread."""
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        """Signal the worker to stop."""
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()

    def _worker(self) -> None:
        while not self._shutdown:
            with self._cond:
                self._cond.wait_for(lambda: bool(self._queue) or self._shutdown, timeout=1.0)
                if self._shutdown:
                    break
                if not self._queue:
                    continue  # timeout — loop to re-check shutdown flag
                key = self._queue.popleft()
                repo, refs_tuple = key
                self.active_repo = repo
                self._active_ref_key = key
            try:
                self._build(repo, list(refs_tuple))
            except Exception:
                log.exception("Build failed for %s", repo)
            finally:
                with self._cond:
                    self.active_repo = None
                    self._active_ref_key = None

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
