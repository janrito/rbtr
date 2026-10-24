"""Run the application that answers requests and keeps watched code indexed."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import signal
import threading
import time
from functools import partial
from pathlib import Path
from typing import assert_never

import structlog
import zmq
import zmq.asyncio
from pydantic import BaseModel

from rbtr import get_version
from rbtr.config import config
from rbtr.daemon.handlers import (
    handle_changed_symbols,
    handle_daemon_config,
    handle_find_refs,
    handle_forget,
    handle_forget_stale,
    handle_gc,
    handle_gc_all,
    handle_list_symbols,
    handle_read_symbol,
    handle_search,
    handle_search_all,
    handle_status,
    handle_status_all,
    handle_unwatch,
    handle_unwatch_stale,
    handle_unwatch_stale_all,
    handle_watch,
)
from rbtr.daemon.messages import (
    ActiveJob,
    AutoRebuildNotification,
    EmbedEndedNotification,
    ErrorCode,
    ErrorResponse,
    GcAllRequest,
    GcEndedNotification,
    GcProgressNotification,
    GcRequest,
    GcStartedNotification,
    IndexErrorNotification,
    Notification,
    ProgressNotification,
    ReadyNotification,
    Response,
)
from rbtr.daemon.rpc import RPC, Handler, RpcResource
from rbtr.daemon.status import remove_status, write_status
from rbtr.errors import IndexNotBuiltError, MissingSnapshot
from rbtr.index import watcher
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index
from rbtr.index.indexer import (
    BuildJob,
    BuildReady,
    CollectionPhase,
    CollectionProgress,
    EmbeddingEnded,
    Indexer,
    IndexerOutput,
    IndexingError,
    RunningJob,
)
from rbtr.index.reranker import Reranker
from rbtr.languages.manager import get_manager

log = structlog.get_logger(__name__)


def _notify(sock: zmq.Socket, notification: BaseModel) -> None:
    """Serialise a notification model and send it over a zmq socket."""
    sock.send(notification.model_dump_json().encode())


class Daemon:
    """Serve code requests and background indexing for a caller-owned Index.

    Construction synchronously seeds HEAD watches. Give it exclusive writable
    access, before concurrent use or while holding the writer semaphore, off a
    UI loop. With `index=None`, only configuration and shutdown are available.

    Use `serve` once on one event loop. Stop through `request_shutdown` rather
    than cancellation, then await serving before closing the supplied Index.
    Daemon owns its models and transport and closes them during teardown.
    """

    def __init__(
        self,
        runtime_dir: Path,
        index: Index | None = None,
        *,
        idle_poll_interval: float | None = None,
        busy_poll_interval: float | None = None,
        allow_missing_plugins: bool = False,
    ) -> None:
        """Prepare runtime paths and centrally configured model handles.

        `runtime_dir` holds discovery and sockets; poll intervals override config.
        `allow_missing_plugins` permits serving data with unavailable language
        plugins; otherwise serving rejects it before readiness.
        """
        self.runtime_dir = runtime_dir
        self.rpc_addr = f"ipc://{runtime_dir / 'daemon.rpc'}"
        self.pub_addr = f"ipc://{runtime_dir / 'daemon.pub'}"
        self._zmq_ctx = zmq.asyncio.Context()
        self._zmq_shadow: zmq.Context = zmq.Context.shadow(self._zmq_ctx)
        handlers = {"daemon_config": Handler(RpcResource.READ, handle_daemon_config)}
        self._allow_missing_plugins = allow_missing_plugins
        self._index = index
        self._ready = threading.Event()
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None

        self._write_sem = asyncio.Semaphore(1)
        self._indexer: Indexer | None = None
        self._warmup = config.warmup
        # Serialises all GPU inference (search + embed batches
        # + idle-unload) so Metal is never driven concurrently.
        self._gpu_lock = asyncio.Lock()
        # Serialises model loading across slots so two concurrent
        # ggml_metal_init calls cannot saturate the GPU.
        self._gpu_load_lock = threading.Lock()
        # Set on shutdown, so an embedding batch in progress ends at once
        # and a build ends before its next file.
        self._stopping = threading.Event()

        if index is not None:
            if config.embed:
                self._embedder = Embedder(
                    idle_timeout=config.embed_idle_timeout,
                    gpu_lock=self._gpu_lock,
                    load_lock=self._gpu_load_lock,
                    abort=self._stopping,
                )
            if config.reranker_model:
                self._reranker = Reranker(
                    idle_timeout=config.reranker_idle_timeout,
                    gpu_lock=self._gpu_lock,
                    load_lock=self._gpu_load_lock,
                )
            self._indexer = Indexer(
                index,
                embedder=self._embedder,
                gpu_lock=self._gpu_lock,
                write_sem=self._write_sem,
                stop=self._stopping,
                on_output=self._publish_index_output,
                idle_poll_interval=idle_poll_interval,
                busy_poll_interval=busy_poll_interval,
            )
            handlers.update(self._index_handlers(index))
        self._rpc = RPC(
            self._zmq_ctx,
            self.rpc_addr,
            handlers=handlers,
            gpu_lock=self._gpu_lock,
            write_sem=self._write_sem,
            index_error=self._index_error,
            on_shutdown=self.request_shutdown,
            on_failure=self._stop_when_it_fails,
        )

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

    def _index_handlers(self, index: Index) -> dict[str, Handler]:
        """Bind operations over `index` to their request kinds and resource classes."""
        return {
            "search": Handler(
                RpcResource.GPU,
                partial(
                    handle_search, index=index, embedder=self._embedder, reranker=self._reranker
                ),
            ),
            "search_all": Handler(
                RpcResource.GPU,
                partial(
                    handle_search_all,
                    index=index,
                    embedder=self._embedder,
                    reranker=self._reranker,
                ),
            ),
            "read_symbol": Handler(RpcResource.READ, partial(handle_read_symbol, index=index)),
            "list_symbols": Handler(RpcResource.READ, partial(handle_list_symbols, index=index)),
            "find_refs": Handler(RpcResource.READ, partial(handle_find_refs, index=index)),
            "changed_symbols": Handler(
                RpcResource.READ, partial(handle_changed_symbols, index=index)
            ),
            "status": Handler(
                RpcResource.READ,
                partial(handle_status, index=index, snapshot_status=self._snapshot_status),
            ),
            "status_all": Handler(
                RpcResource.READ,
                partial(handle_status_all, index=index, snapshot_status=self._snapshot_status),
            ),
            "gc": Handler(RpcResource.WRITE, partial(self._collect, index=index)),
            "gc_all": Handler(RpcResource.WRITE, partial(self._collect, index=index)),
            "unwatch": Handler(RpcResource.WRITE, partial(handle_unwatch, index=index)),
            "unwatch_stale": Handler(RpcResource.WRITE, partial(handle_unwatch_stale, index=index)),
            "unwatch_stale_all": Handler(
                RpcResource.WRITE, partial(handle_unwatch_stale_all, index=index)
            ),
            "forget": Handler(RpcResource.WRITE, partial(handle_forget, index=index)),
            "forget_stale": Handler(RpcResource.WRITE, partial(handle_forget_stale, index=index)),
            "watch": Handler(
                RpcResource.WRITE, partial(handle_watch, index=index), self._watch_result
            ),
        }

    def _collect(self, request: GcRequest | GcAllRequest, index: Index) -> Response:
        """Run foreground collection with a thread-owned notification socket."""
        push = self._zmq_shadow.socket(zmq.PUSH)
        push.connect("inproc://progress")
        _notify(push, GcStartedNotification())
        try:
            match request:
                case GcRequest():
                    return handle_gc(request, index, allow_compact=True)
                case GcAllRequest():
                    return handle_gc_all(
                        request,
                        index,
                        allow_compact=True,
                        on_progress=lambda _repo_path, done, total: _notify(
                            push, GcProgressNotification(current=done, total=total)
                        ),
                    )
        finally:
            _notify(push, GcEndedNotification())
            push.close()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """Wait up to `timeout` seconds for bound sockets and published discovery.

        Return `True` when ready, else `False`. Call outside the serving loop.
        Model loading and indexing may still be pending.
        """
        return self._ready.wait(timeout=timeout)

    def _will_index(self, missing: MissingSnapshot | None) -> bool:
        """Whether the daemon will index what a read found missing, unasked.

        It will when it is building that commit of that repo, or when the
        repo watches the ref the read named. With no snapshot named, the
        index's tables were missing, and any build creates them.  Called
        from `to_thread`: it reads the store.
        """
        run = self._indexer.snapshot() if self._indexer is not None else None
        active = run.work if run is not None and isinstance(run.work, BuildJob) else None
        if missing is None:
            return active is not None
        # A snapshot was named, so a handler read the store: it is open.
        index = self._index
        if index is None:
            return False
        building_it = (
            active is not None
            and missing.snapshot_sha in active.refs
            and active.repo_path == missing.repo_path
        )
        if building_it:
            return True
        repo = index.find_repo(missing.repo_path)
        return repo is not None and missing.ref in index.watched_refs(repo)

    def _snapshot_status(
        self,
        path: str | None,
    ) -> tuple[ActiveJob | None, ActiveJob | None]:
        """Return detached build/embed status from one run, filtered by repository."""
        run = self._indexer.snapshot() if self._indexer is not None else None
        if run is None or path not in (None, run.work.repo_path):
            return None, None
        active = ActiveJob(
            repo_path=run.work.repo_path,
            ref="" if isinstance(run.work, BuildJob) else run.work.at.snapshot_sha,
            phase=run.phase,
            current=run.current,
            total=run.total,
            elapsed_seconds=time.monotonic() - run.started_at,
        )
        return (active, None) if isinstance(run.work, BuildJob) else (None, active)

    def request_shutdown(self) -> None:
        """Request stop without waiting; safe to call repeatedly from any thread."""
        self._rpc.request_shutdown()
        self._stopping.set()
        if self._indexer is not None:
            self._indexer.wake()

    def _watch_result(self, response: Response) -> None:
        """Wake selection for a returned successful watch response, on the loop."""
        if not isinstance(response, ErrorResponse) and self._indexer is not None:
            self._indexer.wake()

    def _index_error(self, error: IndexNotBuiltError) -> ErrorResponse:
        """Map a missing target to an in-progress or not-built response, off the loop."""
        if self._will_index(error.missing):
            return ErrorResponse(
                code=ErrorCode.INDEX_IN_PROGRESS,
                message="Index is building. Try again shortly.",
            )
        return ErrorResponse(code=ErrorCode(error.error_code), message=str(error))

    def _stop_when_it_fails(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (exc := task.exception()) is not None:
            log.error("task_failed", task=task.get_name(), exc_info=exc)
            self.request_shutdown()

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
        """Serve until stop, then finish owned work and dispose of owned resources.

        Return after native work, output, scheduled warmups and cleanup finish;
        a shutdown-request reply can precede this completion.

        Preflight, transport and background-operation failures reach the caller.
        Scheduled warmup errors are suppressed. Individual request/job errors
        use their response/output paths.
        """
        # Refuse before readiness if rebuilding would lose unavailable language data.
        if self._index is not None:
            get_manager().require_languages_loaded(
                self._index.distinct_chunk_languages(),
                allow_missing=self._allow_missing_plugins,
            )
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self._cleanup)
        if self._embedder is not None:
            self._embedder.start_idle_monitor()
        if self._reranker is not None:
            self._reranker.start_idle_monitor()
        self._install_signal_handlers()

        ctx = self._zmq_ctx
        pub: zmq.asyncio.Socket = ctx.socket(zmq.PUB)
        progress_pull: zmq.asyncio.Socket = ctx.socket(zmq.PULL)
        progress_pull.bind("inproc://progress")
        self._notify_push: zmq.asyncio.Socket = ctx.socket(zmq.PUSH)
        self._notify_push.connect("inproc://progress")
        warmup_tasks: list[asyncio.Task[None]] = []
        index_tasks: list[asyncio.Task[None]] = []
        relay_task: asyncio.Task[None] | None = None

        def ready() -> None:
            nonlocal relay_task
            # RPC and PUB have both bound before publishing discovery.
            write_status(
                self.runtime_dir,
                pid=os.getpid(),
                rpc=self.rpc_addr,
                pub=self.pub_addr,
                version=get_version(),
            )
            log.info("daemon_listening", rpc=self.rpc_addr, pub=self.pub_addr)
            self._ready.set()
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
            if self._indexer is not None:
                index_tasks.extend(
                    [
                        asyncio.create_task(self._indexer.run(), name="job_worker"),
                        asyncio.create_task(self._indexer.watch(), name="watcher"),
                        asyncio.create_task(self._indexer.maintain(), name="maintenance"),
                    ]
                )
            for task in index_tasks:
                task.add_done_callback(self._stop_when_it_fails)

        try:
            pub.bind(self.pub_addr)
            await self._rpc.serve(ready)
        finally:
            self.request_shutdown()
            try:
                await asyncio.gather(*index_tasks, return_exceptions=True)
                for task in index_tasks:
                    if not task.cancelled():
                        task.result()
            finally:
                if relay_task is not None:
                    relay_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await relay_task
                # Scheduled native warmups finish even if their executor await is queued.
                for task in warmup_tasks:
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await task
                self._cleanup()
                self._notify_push.close()
                progress_pull.close()
                self._rpc.close()
                pub.close()
                ctx.term()
                log.info("daemon_stopped")

    async def _publish_index_output(self, output: IndexerOutput) -> None:
        """Project typed indexing output onto the existing notification protocol."""
        notification: Notification
        match output:
            case RunningJob():
                notification = ProgressNotification(
                    repo_path=output.work.repo_path,
                    phase=output.phase,
                    current=output.current,
                    total=output.total,
                )
            case BuildReady():
                notification = ReadyNotification(
                    repo_path=output.at.repo.repo_path,
                    ref=output.at.snapshot_sha,
                    chunks=output.counts.total,
                    embedded=output.counts.embedded,
                    edges=output.result.stats.total_edges,
                    elapsed=round(output.result.stats.elapsed_seconds, 2),
                )
            case EmbeddingEnded():
                notification = EmbedEndedNotification(
                    repo_path=output.at.repo.repo_path,
                    ref=output.at.snapshot_sha,
                    chunks=output.counts.total,
                    embedded=output.counts.embedded,
                    outcome=output.outcome,
                )
            case IndexingError():
                notification = IndexErrorNotification(
                    repo_path=output.repo_path, message=output.message
                )
            case watcher.WatchedTarget() | watcher.DirtyWorktree():
                notification = AutoRebuildNotification(
                    repo_path=output.repo_path, new_ref=output.snapshot_sha
                )
            case CollectionPhase.STARTED:
                notification = GcStartedNotification()
            case CollectionPhase.ENDED:
                notification = GcEndedNotification()
            case CollectionProgress():
                notification = GcProgressNotification(current=output.current, total=output.total)
            case _:
                assert_never(output)
        await self._notify_push.send(notification.model_dump_json().encode())

    async def _notification_relay(self, pull: zmq.asyncio.Socket, pub: zmq.asyncio.Socket) -> None:
        """Forward notifications from inproc PULL to PUB."""
        while True:
            await pub.send(await pull.recv())
