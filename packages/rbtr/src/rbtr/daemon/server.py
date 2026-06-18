"""ZMQ daemon server.

Binds two sockets in `runtime_dir`:

- **REP** (`daemon.rpc`) — synchronous request/response.
  Receives a `Request` (discriminated on `kind`), dispatches
  to the matching handler, returns a `Response`.
- **PUB** (`daemon.pub`) — fan-out notifications.
  Broadcasts `Notification` messages (index progress,
  ready, auto-rebuild) to any connected SUB clients (pi-rbtr
  extension, CLI listeners).

The server uses an asyncio.TaskGroup with named tasks:
RPC loop, job worker, watcher, notification relay, and
idle monitor.  Worker threads send progress via zmq
inproc PUSH; an async PULL task receives and forwards
to PUB.

Lifecycle::

    server = DaemonServer(Path.home() / ".rbtr")
    asyncio.run(server.serve())       # blocks until shutdown
    # or from another thread:
    server.request_shutdown()         # thread-safe
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import inspect
import itertools
import json
import os
import signal
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
import zmq
import zmq.asyncio
from pydantic import BaseModel, ValidationError

from rbtr import get_version
from rbtr.config import config
from rbtr.daemon import watcher
from rbtr.daemon.handlers import (
    handle_build_index,
    handle_changed_symbols,
    handle_find_refs,
    handle_gc,
    handle_list_symbols,
    handle_read_symbol,
    handle_search,
    handle_status,
    resolve_refs,
)
from rbtr.daemon.messages import (
    ActiveJob,
    AutoRebuildNotification,
    BuildJob,
    EmbedCompleteNotification,
    EmbedJob,
    ErrorCode,
    ErrorResponse,
    HasRepoPath,
    IndexErrorNotification,
    OkResponse,
    ProgressNotification,
    ReadyNotification,
    Response,
    ShutdownRequest,
    request_adapter,
)
from rbtr.daemon.status import remove_status, write_status
from rbtr.errors import IndexNotBuiltError, RbtrError
from rbtr.git import HEAD_REF, filter_tree_shas, normalise_repo_path
from rbtr.index.embeddings import Embedder, embedding_text
from rbtr.index.orchestrator import ProgressCallback, build_index, embed_index
from rbtr.index.reranker import Reranker
from rbtr.index.store import IndexStore
from rbtr.logging import elapsed_ms

log = structlog.get_logger(__name__)


type RequestHandler = Callable[[Any], Response | Awaitable[Response]]


def _notify(sock: zmq.Socket, notification: BaseModel) -> None:
    """Serialise a notification model and send it over a zmq socket."""
    sock.send(notification.model_dump_json().encode())


def _format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic `ValidationError` as concise per-field feedback.

    Each line names the offending field, what was wrong, and the value
    actually received — enough for a caller (human or model) to see how
    an argument was mis-shaped and resend it correctly, for any field,
    not just one hand-picked failure mode.
    """
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        received = repr(err.get("input"))
        if len(received) > 120:
            received = f"{received[:117]}..."
        lines.append(f"  - {loc}: {err['msg']} (received: {received})")
    body = "\n".join(lines) or "  - (no detail)"
    return f"Invalid request arguments:\n{body}"


def _progress_callback(push: zmq.Socket, path: str) -> ProgressCallback:
    """Return a progress callback that sends `ProgressNotification`s."""

    def on_progress(phase: str, done: int, total: int) -> None:
        _notify(push, ProgressNotification(repo_path=path, phase=phase, current=done, total=total))

    return on_progress


class DaemonServer:
    """ZMQ daemon server managing REP and PUB sockets."""

    def __init__(
        self,
        runtime_dir: Path,
        store: IndexStore | None = None,
        *,
        idle_poll_interval: float | None = None,
        busy_poll_interval: float | None = None,
    ) -> None:
        # Defaults come from the central pydantic Config so there is
        # exactly one source of truth per knob. Callers (currently
        # only tests) may override either interval explicitly.
        idle_poll_interval = (
            idle_poll_interval if idle_poll_interval is not None else config.idle_poll_interval
        )
        busy_poll_interval = (
            busy_poll_interval if busy_poll_interval is not None else config.busy_poll_interval
        )
        self.runtime_dir = runtime_dir
        self.rpc_addr = f"ipc://{runtime_dir / 'daemon.rpc'}"
        self.pub_addr = f"ipc://{runtime_dir / 'daemon.pub'}"
        self._shutdown = False
        self._pub_socket: zmq.asyncio.Socket | None = None
        self._zmq_ctx = zmq.asyncio.Context()
        self._zmq_shadow = zmq.Context.shadow(self._zmq_ctx)
        self._handlers: dict[str, RequestHandler] = {
            "shutdown": self._handle_shutdown,
        }
        self._idle_poll_interval = idle_poll_interval
        self._busy_poll_interval = busy_poll_interval
        self._store = store
        self._ready = threading.Event()
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None

        # Job worker state — event loop thread only.
        self._wake = asyncio.Event()
        self._write_sem = asyncio.Semaphore(1)
        self._active_key: str | None = None
        self._active_build: ActiveJob | None = None
        self._active_embed: ActiveJob | None = None
        self._started_at: float | None = None
        self._warmup = config.warmup
        # Serialises all GPU inference (search + embed batches
        # + idle-unload) so Metal is never driven concurrently.
        self._gpu_lock = asyncio.Lock()
        # Serialises model loading across slots so two concurrent
        # ggml_metal_init calls cannot saturate the GPU.
        self._gpu_load_lock = threading.Lock()

        if store is not None:
            self._embedder = Embedder(
                idle_timeout=config.embed_idle_timeout,
                gpu_lock=self._gpu_lock,
                load_lock=self._gpu_load_lock,
            )
            if config.reranker_model:
                self._reranker = Reranker(
                    idle_timeout=config.reranker_idle_timeout,
                    gpu_lock=self._gpu_lock,
                    load_lock=self._gpu_load_lock,
                )
            self._register_index_handlers(store)
            self._backfill_head_watches(store)
            self._recover_pending_embeds(store)

    def _register_atexit(self) -> None:
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        remove_status(self.runtime_dir)
        (self.runtime_dir / "daemon.rpc").unlink(missing_ok=True)
        (self.runtime_dir / "daemon.pub").unlink(missing_ok=True)
        if self._embedder is not None:
            try:
                self._embedder.close()
            except Exception:
                log.exception("embedder_cleanup_failed")
        if self._reranker is not None:
            try:
                self._reranker.close()
            except Exception:
                log.exception("reranker_cleanup_failed")

    def _register_index_handlers(self, store: IndexStore) -> None:
        emb = self._embedder
        rnk = self._reranker

        async def _async_search(req: Any) -> Response:
            async with self._gpu_lock:
                return await asyncio.to_thread(
                    handle_search,
                    req,
                    store,
                    embedder=emb,
                    reranker=rnk,
                )

        self._handlers.update(
            {
                "search": _async_search,
                "read_symbol": lambda req: handle_read_symbol(req, store),
                "list_symbols": lambda req: handle_list_symbols(req, store),
                "find_refs": lambda req: handle_find_refs(req, store),
                "changed_symbols": lambda req: handle_changed_symbols(req, store),
                "status": lambda req: handle_status(
                    req,
                    store,
                    self._snapshot_status,
                ),
                "gc": lambda req: handle_gc(req, store),
                "index": lambda req: handle_build_index(req, self.watch_refs),
            }
        )

    def _backfill_head_watches(self, store: IndexStore) -> None:
        """Seed a `"HEAD"` watch for every registered repo (idempotent).

        Ensures HEAD tracking for repos registered before the
        `watched_refs` table existed; new repos get their HEAD row
        through the `index` request path.
        """
        repos = store.list_repos()
        if not repos:
            return
        with store.session() as ws:
            for repo_id, _repo_path in repos:
                ws.add_watched_refs(repo_id, [HEAD_REF])
        log.info("head_watch_backfill", repos=len(repos))

    def _recover_pending_embeds(self, store: IndexStore) -> None:
        """Check for indexed commits with un-embedded chunks and wake the worker.

        Handles the case where the daemon crashed after indexing
        but before embedding completed.  Sets `_wake` so the
        DB-polling worker picks up the work.
        """
        for repo_id, _repo_path in store.list_repos():
            for sha, _ts in store.list_indexed_commits(repo_id):
                count = store.count_unembedded(repo_id, sha)
                if count > 0:
                    log.info(
                        "recovering_embed",
                        repo_id=repo_id,
                        sha=sha[:12],
                        chunks=count,
                    )
                    self._wake.set()
                    return

    def _run_build(self, job: BuildJob, store: IndexStore, push: zmq.Socket) -> None:
        """Execute a build job.  Called from `_run_job`."""
        with store.session() as ws:
            repo_id = ws.register_repo(job.repo_path)

        try:
            resolved = resolve_refs(job.repo_path, list(job.refs))
        except RbtrError as exc:
            _notify(push, IndexErrorNotification(repo_path=job.repo_path, message=str(exc)))
            return

        for sha in resolved:
            result = build_index(
                job.repo_path,
                sha,
                store,
                repo_id=repo_id,
                on_progress=_progress_callback(push, job.repo_path),
            )
            total = store.count_chunks(sha, repo_id=repo_id)
            unembedded = store.count_unembedded(repo_id, sha)
            _notify(
                push,
                ReadyNotification(
                    repo_path=job.repo_path,
                    ref=sha,
                    chunks=total,
                    embedded=total - unembedded,
                    edges=result.stats.total_edges,
                    elapsed=round(result.stats.elapsed_seconds, 2),
                ),
            )
            # Clean up stale worktree tree SHAs.  After a
            # worktree build, old tree SHAs from previous edits
            # linger in indexed_commits.  After a commit build,
            # the old worktree was built against the previous
            # HEAD's tree.  Either way, drop any tree-type SHA
            # that isn't the one we just built.
            self._drop_stale_worktree_shas(store, repo_id, job.repo_path, keep=sha)

    @staticmethod
    def _drop_stale_worktree_shas(
        store: IndexStore,
        repo_id: int,
        repo_path: str,
        *,
        keep: str,
    ) -> None:
        """Drop old worktree tree SHAs from `indexed_commits`.

        After a build (commit or worktree), scans `indexed_commits`
        for tree-type SHAs and drops any that aren't *keep*.  This
        prevents stale worktree rows from accumulating between GC
        runs.

        Called from `_run_build` inside the worker thread's
        `WriteSession` scope.
        """
        indexed = [sha for sha, _ts in store.list_indexed_commits(repo_id)]
        stale = [sha for sha in filter_tree_shas(repo_path, indexed) if sha != keep]
        if not stale:
            return
        with store.session() as ws:
            for sha in stale:
                ws.drop_commit(repo_id, sha)
                log.info("dropped_stale_worktree_sha", sha=sha[:12])

    def _run_embed(self, job: EmbedJob, store: IndexStore, push: zmq.Socket) -> None:
        """Execute an embed job.  Called from `_run_job`.

        Yields between batches when a higher-priority build is
        pending.  Remaining chunks are still unembedded so the
        next call resumes where this one left off.
        """
        if self._embedder is None:
            return

        embed_index(
            store,
            job.ref,
            repo_id=job.repo_id,
            embedder=self._embedder,
            on_progress=_progress_callback(push, job.repo_path),
            should_stop=lambda: bool(watcher.poll_watched(store)),
        )
        total = store.count_chunks(job.ref, repo_id=job.repo_id)
        unembedded = store.count_unembedded(job.repo_id, job.ref)
        _notify(
            push,
            EmbedCompleteNotification(
                repo_path=job.repo_path,
                ref=job.ref,
                chunks=total,
                embedded=total - unembedded,
            ),
        )

    async def _run_embed_async(self, job: EmbedJob) -> None:
        """Async embed runner — acquires `_gpu_lock` per-batch.

        Replaces monolithic `to_thread(_run_embed)` so search
        requests wait at most one batch duration.
        """
        store = self._store
        embedder = self._embedder
        if store is None or embedder is None:
            return

        total = await asyncio.to_thread(store.count_unembedded, job.repo_id, job.ref)
        if total == 0:
            return

        push = self._zmq_shadow.socket(zmq.PUSH)
        push.connect("inproc://progress")
        on_progress = _progress_callback(push, job.repo_path)
        on_progress("loading_model", 0, 0)
        done = 0
        t0 = time.perf_counter()

        try:
            while True:
                missing = await asyncio.to_thread(store.get_unembedded_chunks, job.repo_id, job.ref)
                if not missing:
                    break
                before = done
                for batch in itertools.batched(missing, config.embedding_batch_size, strict=False):
                    texts = [embedding_text(c.name, c.content) for c in batch]
                    try:
                        async with self._gpu_lock:
                            results = await asyncio.to_thread(embedder.embed, texts)
                    except (RuntimeError, ValueError):
                        log.warning("embedding_batch_failed", exc_info=True)
                        continue
                    await asyncio.to_thread(
                        self._write_embed_batch,
                        store,
                        job.repo_id,
                        batch,
                        [r.vector for r in results],
                        [r.truncated for r in results],
                    )
                    done += len(batch)
                    on_progress("embedding", done, total)
                    if self._shutdown:
                        log.info("embedding_stopped_shutdown", done=done, total=total)
                        return
                    # Yield to higher-priority builds or worktree rebuilds.
                    stale = await asyncio.to_thread(watcher.poll_watched, store)
                    dirty = await asyncio.to_thread(watcher.poll_worktree, store)
                    if stale or dirty:
                        log.info("embedding_preempted", done=done, total=total)
                        return
                if done == before:
                    break
        finally:
            total_chunks = await asyncio.to_thread(
                store.count_chunks,
                job.ref,
                repo_id=job.repo_id,
            )
            unembedded = await asyncio.to_thread(
                store.count_unembedded,
                job.repo_id,
                job.ref,
            )
            _notify(
                push,
                EmbedCompleteNotification(
                    repo_path=job.repo_path,
                    ref=job.ref,
                    chunks=total_chunks,
                    embedded=total_chunks - unembedded,
                ),
            )
            push.close()
        log.info("embedded_chunks", done=done, total=total, elapsed_ms=elapsed_ms(t0))

    @staticmethod
    def _write_embed_batch(
        store: IndexStore,
        repo_id: int,
        batch: tuple[Any, ...],
        vectors: list[list[float]],
        truncated: list[bool] | None = None,
    ) -> None:
        """Write one embedding batch in its own session."""
        with store.session() as session:
            session.update_embeddings(
                [c.id for c in batch], vectors, repo_id=repo_id, truncated=truncated
            )

    def register(self, kind: str, handler: RequestHandler) -> None:
        self._handlers[kind] = handler

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """Block until the server is accepting requests.

        Returns True if ready, False if timed out.
        """
        return self._ready.wait(timeout=timeout)

    def watch_refs(self, repo_path: str, refs: list[str], *, remove: bool) -> None:
        """Add or remove watched refs for a repo, then wake the worker.

        Records intent in `watched_refs`; `poll_watched` derives the
        actual build on its next poll.  On *remove*, `"HEAD"` is
        rejected with `RbtrError` **before any delete** so the whole
        request fails atomically — HEAD is the default always-watched
        ref.
        """
        if self._store is None:
            return
        if remove:
            if HEAD_REF in refs:
                msg = "HEAD cannot be removed from the watch set"
                raise RbtrError(msg)
            repo_id = self._store.get_repo_id(repo_path)
            if repo_id is None:
                return  # nothing watched for an unregistered repo
            with self._store.session() as ws:
                ws.remove_watched_refs(repo_id, refs)
            log.info("watched_refs_removed", repo=repo_path, refs=refs)
            return
        with self._store.session() as ws:
            repo_id = ws.register_repo(repo_path)
            # HEAD is always watched and cannot be removed; ensure it
            # here so any `index` (not just startup backfill) upholds it.
            ws.add_watched_refs(repo_id, [HEAD_REF, *refs])
        log.info("watched_refs_added", repo=repo_path, refs=refs)
        self._wake.set()

    def _is_building(self) -> bool:
        """Return True if a build is currently active."""
        return self._active_build is not None

    def _snapshot_status(
        self,
        path: str,
    ) -> tuple[ActiveJob | None, ActiveJob | None]:
        """Return `(active_build, active_embed, pending_builds)`.

        Only returns jobs whose `path` matches so a repo-scoped
        status query doesn't leak activity from unrelated repos.
        """
        active_build: ActiveJob | None = None
        if self._active_build is not None and self._started_at is not None:
            job = self._active_build.model_copy(
                update={"elapsed_seconds": time.monotonic() - self._started_at}
            )
            if job.repo_path == path:
                active_build = job
        active_embed: ActiveJob | None = None
        if self._active_embed is not None and self._started_at is not None:
            job = self._active_embed.model_copy(
                update={"elapsed_seconds": time.monotonic() - self._started_at}
            )
            if job.repo_path == path:
                active_embed = job
        return active_build, active_embed

    def request_shutdown(self) -> None:
        self._shutdown = True
        self._wake.set()

    # ── DB-polling worker ─────────────────────────────────────────────────

    def _find_next_job(self) -> BuildJob | EmbedJob | None:
        """Query the DB for the next piece of work.

        Priority: un-indexed commits (builds) before un-embedded
        chunks (embeds).  Skips the repo/ref that is currently
        active.  Runs in the event loop thread (fast DuckDB read).
        """
        store = self._store
        if store is None:
            return None

        # Builds: un-indexed watched refs (HEAD is the default one).
        for target in watcher.poll_watched(store):
            key = target.repo_path
            if key == self._active_key:
                continue
            return BuildJob(repo_path=target.repo_path, refs=(target.sha,))

        # Worktree builds: dirty working trees.
        for dirty in watcher.poll_worktree(store):
            key = f"{dirty.repo_path}:wt:{dirty.tree_sha}"
            if key == self._active_key:
                continue
            return BuildJob(repo_path=dirty.repo_path, refs=(dirty.tree_sha,))

        # Embeds: indexed commits with un-embedded chunks.
        for repo_id, repo_path in store.list_repos():
            for sha, _ts in store.list_indexed_commits(repo_id):
                count = store.count_unembedded(repo_id, sha)
                if count > 0:
                    key = f"{repo_id}:{sha}"
                    if key == self._active_key:
                        continue
                    return EmbedJob(repo_path=repo_path, repo_id=repo_id, ref=sha)

        return None

    def _set_active(self, job: BuildJob | EmbedJob) -> None:
        """Mark a job as active.  Event loop thread only."""
        self._started_at = time.monotonic()
        match job:
            case BuildJob():
                self._active_key = job.repo_path
                self._active_build = ActiveJob(
                    repo_path=job.repo_path,
                    ref="",
                    phase="starting",
                    current=0,
                    total=0,
                    elapsed_seconds=0.0,
                )
            case EmbedJob():
                self._active_key = f"{job.repo_id}:{job.ref}"
                self._active_embed = ActiveJob(
                    repo_path=job.repo_path,
                    ref=job.ref,
                    phase="embedding",
                    current=0,
                    total=0,
                    elapsed_seconds=0.0,
                )

    def _clear_active(self) -> None:
        """Clear active-job tracking.  Event loop thread only."""
        self._active_key = None
        self._active_build = None
        self._active_embed = None
        self._started_at = None

    def _run_job(self, job: BuildJob | EmbedJob) -> None:
        """Dispatch a job to the appropriate runner.  Called from `to_thread`.

        Owns the inproc PUSH socket for the duration of the job.
        """
        store = self._store
        if store is None:
            return
        push = self._zmq_shadow.socket(zmq.PUSH)
        push.connect("inproc://progress")
        try:
            match job:
                case BuildJob():
                    self._run_build(job, store, push)
                case EmbedJob():
                    self._run_embed(job, store, push)
        finally:
            push.close()

    async def _job_worker(self) -> None:
        """Async task that polls the DB for work and runs jobs.

        Waits on `_wake`, queries the DB via `_find_next_job`,
        and runs each job via `asyncio.to_thread`.  Builds before
        embeds (query ordering).  After a build completes, the
        worker re-checks for embed work before sleeping.

        Embed jobs use `_run_embed_async` which acquires
        `_gpu_lock` per-batch, keeping search responsive.
        Build jobs use the existing monolithic `to_thread` path.
        """
        while not self._shutdown:
            await self._wake.wait()
            self._wake.clear()
            while not self._shutdown:
                job = self._find_next_job()
                if job is None:
                    break
                self._set_active(job)
                job_ctx: dict[str, str] = {
                    "job_id": uuid4().hex[:8],
                    "repo": job.repo_path,
                    "job_kind": "build" if isinstance(job, BuildJob) else "embed",
                }
                if isinstance(job, EmbedJob):
                    job_ctx["ref"] = job.ref
                else:
                    job_ctx["ref"] = job.refs[0][:12]
                # Bound for the job's lifetime; `to_thread` copies the
                # context, so build/embed logs in worker threads carry it.
                with structlog.contextvars.bound_contextvars(**job_ctx):
                    try:
                        if isinstance(job, EmbedJob):
                            await self._run_embed_async(job)
                        else:
                            async with self._write_sem:
                                await asyncio.to_thread(self._run_job, job)
                    except Exception:
                        log.exception("job_failed", job=str(job))
                    finally:
                        self._clear_active()

    def _install_signal_handlers(self) -> None:
        """Register signal handlers via the running event loop.

        Uses `loop.add_signal_handler` which is asyncio-native.
        Only works in the main thread; silently skipped otherwise
        (tests run `serve()` in a worker thread).
        """
        if threading.current_thread() is not threading.main_thread():
            return
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.request_shutdown)

    async def serve(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._register_atexit()

        # Start idle-unload monitors for GPU models.
        if self._embedder is not None:
            self._embedder.start_idle_monitor()
        if self._reranker is not None:
            self._reranker.start_idle_monitor()
        self._install_signal_handlers()

        ctx = self._zmq_ctx
        rep: zmq.asyncio.Socket = ctx.socket(zmq.REP)
        pub: zmq.asyncio.Socket = ctx.socket(zmq.PUB)
        self._pub_socket = pub

        # inproc PULL receives notifications from worker threads and the
        # event-loop PUSH socket, then forwards to PUB.
        progress_pull: zmq.asyncio.Socket = ctx.socket(zmq.PULL)
        progress_pull.bind("inproc://progress")

        # Sync PUSH for event-loop-thread sends (watcher, recovery).
        self._notify_push: zmq.Socket = self._zmq_shadow.socket(zmq.PUSH)
        self._notify_push.connect("inproc://progress")

        try:
            rep.bind(self.rpc_addr)
            pub.bind(self.pub_addr)

            # Write status file only after sockets are bound so clients
            # never see a file with stale endpoints.
            write_status(
                self.runtime_dir,
                pid=os.getpid(),
                rpc=self.rpc_addr,
                pub=self.pub_addr,
                version=get_version(),
            )
            log.info("daemon_listening", rpc=self.rpc_addr, pub=self.pub_addr)
            self._ready.set()

            warmup_tasks: list[asyncio.Task[None]] = []
            if self._warmup:
                if self._embedder is not None:
                    warmup_tasks.append(
                        asyncio.create_task(asyncio.to_thread(self._embedder.warmup))
                    )
                if self._reranker is not None:
                    warmup_tasks.append(
                        asyncio.create_task(asyncio.to_thread(self._reranker.warmup))
                    )
            relay_task = asyncio.create_task(self._notification_relay(progress_pull, pub))
            worker_task = asyncio.create_task(self._job_worker())
            watcher_task = asyncio.create_task(self._watcher_loop())
            try:
                while not self._shutdown:
                    if await rep.poll(timeout=100):
                        raw = await rep.recv()
                        response = await self._dispatch(raw)
                        await rep.send(response.model_dump_json().encode())
            finally:
                # Ensure worker sees shutdown.
                self._shutdown = True
                self._wake.set()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker_task
                watcher_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher_task
                relay_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await relay_task
                for wt in warmup_tasks:
                    wt.cancel()
                    with contextlib.suppress(Exception):
                        await wt
        finally:
            self._cleanup()
            self._pub_socket = None
            self._notify_push.close()
            progress_pull.close()
            rep.close()
            pub.close()
            ctx.term()
            log.info("daemon_stopped")

    async def _notification_relay(self, pull: zmq.asyncio.Socket, pub: zmq.asyncio.Socket) -> None:
        """Receive notifications from inproc PULL and forward to PUB.

        Also updates active-job progress fields so status queries
        reflect the latest state.
        """
        while True:
            raw = await pull.recv()
            # Update active-job progress from ProgressNotification.
            self._update_active_from_notification(raw)
            await pub.send(raw)

    def _update_active_from_notification(self, raw: bytes) -> None:
        """Parse a notification and update active-job progress."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        kind = data.get("kind")
        if kind != "progress":
            return
        phase = data.get("phase", "")
        current = data.get("current", 0)
        total = data.get("total", 0)
        if self._active_build is not None:
            self._active_build = self._active_build.model_copy(
                update={"phase": phase, "current": current, "total": total}
            )
        if self._active_embed is not None:
            self._active_embed = self._active_embed.model_copy(
                update={"phase": phase, "current": current, "total": total}
            )

    def _next_poll_interval(self) -> float:
        """Pick the watcher poll interval based on worker state.

        While a job is active, slow the watcher down so a long
        embed phase doesn't flood the queue with duplicates for
        the same stale SHA. Other repos are still detected on the
        busy cadence — no repo is starved.
        """
        if self._active_key is not None:
            return self._busy_poll_interval
        return self._idle_poll_interval

    async def _watcher_loop(self) -> None:
        """Async task that polls watched refs and dirty worktrees for index work."""
        while not self._shutdown:
            await asyncio.sleep(self._next_poll_interval())
            if self._shutdown:
                break
            if self._store is None:
                continue
            with structlog.contextvars.bound_contextvars(watch_cycle=uuid4().hex[:8]):
                stale_list = await asyncio.to_thread(watcher.poll_watched, self._store)
                for target in stale_list:
                    log.info(
                        "watched_ref_stale",
                        repo=target.repo_path,
                        ref=target.ref,
                        sha=target.sha[:12],
                    )
                    _notify(
                        self._notify_push,
                        AutoRebuildNotification(repo_path=target.repo_path, new_ref=target.sha),
                    )
                    self._wake.set()
                dirty_list = await asyncio.to_thread(watcher.poll_worktree, self._store)
                for dirty in dirty_list:
                    log.info("dirty_worktree", repo=dirty.repo_path, tree=dirty.tree_sha[:12])
                    _notify(
                        self._notify_push,
                        AutoRebuildNotification(repo_path=dirty.repo_path, new_ref=dirty.tree_sha),
                    )
                    self._wake.set()

    async def _dispatch(self, raw: bytes) -> Response:
        try:
            request = request_adapter.validate_json(raw)
        except ValidationError as exc:
            return ErrorResponse(
                code=ErrorCode.INVALID_REQUEST,
                message=_format_validation_error(exc),
            )
        if isinstance(request, HasRepoPath):
            try:
                request.repo_path = normalise_repo_path(request.repo_path)
            except RbtrError as exc:
                # A bad repo_path (e.g. a client whose cwd is not a git
                # repo) must not take the whole daemon down: reply with an
                # error so the surviving daemon keeps serving other repos.
                log.warning("request_rejected", kind=request.kind, error=str(exc))
                return ErrorResponse(code=ErrorCode.REPO_NOT_FOUND, message=str(exc))
        # Bind correlation context for this request.  All keys reset on
        # exit, so no state leaks to the next request.
        req_ctx: dict[str, str] = {"request_id": uuid4().hex[:8], "kind": request.kind}
        if isinstance(request, HasRepoPath):
            req_ctx["repo"] = request.repo_path
        with structlog.contextvars.bound_contextvars(**req_ctx):
            handler = self._handlers.get(request.kind)
            if handler is None:
                return ErrorResponse(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"No handler for kind: {request.kind}",
                )
            t0 = time.perf_counter()
            try:
                result = handler(request)
                if inspect.isawaitable(result):
                    result = await result
            except IndexNotBuiltError as exc:
                # A ref/symbol that isn't indexed yet: if a build is
                # running it will become queryable soon, so say so rather
                # than tell the caller to start an index that is already
                # in progress.  This is the daemon's single source of
                # build-awareness; handlers stay build-agnostic.
                if self._is_building():
                    return ErrorResponse(
                        code=ErrorCode.INDEX_IN_PROGRESS,
                        message="Index is building. Try again shortly.",
                    )
                log.warning("handler_error", error=str(exc), elapsed_ms=elapsed_ms(t0))
                return ErrorResponse(code=ErrorCode(exc.error_code), message=str(exc))
            except RbtrError as exc:
                log.warning("handler_error", error=str(exc), elapsed_ms=elapsed_ms(t0))
                return ErrorResponse(code=ErrorCode(exc.error_code), message=str(exc))
            except Exception as exc:
                log.exception("handler_error", elapsed_ms=elapsed_ms(t0))
                return ErrorResponse(code=ErrorCode.INTERNAL, message=str(exc))
            log.info("request_complete", elapsed_ms=elapsed_ms(t0))
            return result

    def _handle_shutdown(self, _request: ShutdownRequest) -> OkResponse:
        self.request_shutdown()
        return OkResponse()
