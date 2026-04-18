"""ZMQ daemon server.

Binds two sockets in `sock_dir`:

- **REP** (`daemon.rpc`) — synchronous request/response.
  Receives a `Request` (discriminated on `kind`), dispatches
  to the matching handler, returns a `Response`.
- **PUB** (`daemon.pub`) — fan-out notifications.
  Broadcasts `Notification` messages (index progress,
  ready, auto-rebuild) to any connected SUB clients (pi-rbtr
  extension, CLI listeners).

The event loop polls REP with a 100 ms timeout. Between polls
it drains a notification queue fed by build threads and the
ref watcher.

Lifecycle::

    server = DaemonServer(Path.home() / ".rbtr")
    anyio.run(server.serve)           # blocks until shutdown
    # or from another thread:
    server.request_shutdown()         # thread-safe
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import zmq
import zmq.asyncio

from rbtr import get_version
from rbtr.config import config
from rbtr.daemon.build_queue import BuildQueue
from rbtr.daemon.handlers import (
    handle_build_index,
    handle_changed_symbols,
    handle_find_refs,
    handle_gc,
    handle_list_symbols,
    handle_read_symbol,
    handle_search,
    handle_status,
)
from rbtr.daemon.messages import (
    AutoRebuildNotification,
    ErrorCode,
    ErrorResponse,
    Notification,
    OkResponse,

    Request,
    Response,
    ShutdownRequest,
    request_adapter,
)
from rbtr.daemon import watcher
from rbtr.daemon.repos import RepoManager
from rbtr.daemon.status import remove_status, write_status
from rbtr.index.store import IndexStore

log = logging.getLogger(__name__)

type RequestHandler = Callable[[Any], Response]


class DaemonServer:
    """ZMQ daemon server managing REP and PUB sockets."""

    def __init__(
        self,
        sock_dir: Path,
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
        self.sock_dir = sock_dir
        self.rpc_addr = f"ipc://{sock_dir / 'daemon.rpc'}"
        self.pub_addr = f"ipc://{sock_dir / 'daemon.pub'}"
        self._shutdown = False
        self._pub_socket: zmq.asyncio.Socket | None = None
        self._notification_queue: queue.SimpleQueue[Notification] = queue.SimpleQueue()
        self._handlers: dict[str, RequestHandler] = {
            "shutdown": self._handle_shutdown,
        }
        self._build_queue: BuildQueue | None = None
        self._idle_poll_interval = idle_poll_interval
        self._busy_poll_interval = busy_poll_interval
        self._store = store
        if store is not None:
            self._register_index_handlers(store)

    def _register_atexit(self) -> None:
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        remove_status(self.sock_dir)
        (self.sock_dir / "daemon.rpc").unlink(missing_ok=True)
        (self.sock_dir / "daemon.pub").unlink(missing_ok=True)

    def _register_index_handlers(self, store: IndexStore) -> None:
        mgr = RepoManager(store)
        bq = BuildQueue(mgr, self._notification_queue.put)
        self._build_queue = bq
        self._handlers.update(
            {
                "search": lambda req: handle_search(req, mgr),
                "read_symbol": lambda req: handle_read_symbol(req, mgr),
                "list_symbols": lambda req: handle_list_symbols(req, mgr),
                "find_refs": lambda req: handle_find_refs(req, mgr),
                "changed_symbols": lambda req: handle_changed_symbols(req, mgr),
                "status": lambda req: handle_status(req, mgr, bq),
                "gc": lambda req: handle_gc(req, mgr),
                "index": lambda req: handle_build_index(req, bq),
            }
        )

    def register(self, kind: str, handler: RequestHandler) -> None:
        self._handlers[kind] = handler

    def request_shutdown(self) -> None:
        self._shutdown = True
        if self._build_queue is not None:
            self._build_queue.stop()

    def _setup_signal_handlers(self) -> None:
        # Only set up signal handlers in the main thread. In tests
        # the server may run in a worker thread where signal.signal
        # raises ValueError.
        try:
            import threading
        except ImportError:
            return
        if threading.current_thread() is not threading.main_thread():
            return

        def _handle_shutdown_signal(signum: int, _frame: Any) -> None:
            log.info("Received signal %d, shutting down", signum)
            self.request_shutdown()

        signal.signal(signal.SIGTERM, _handle_shutdown_signal)
        signal.signal(signal.SIGINT, _handle_shutdown_signal)

    async def serve(self) -> None:
        self.sock_dir.mkdir(parents=True, exist_ok=True)
        self._register_atexit()

        # Start the embedding idle-unload monitor if a store was provided
        # (store init loads the model, so we track from here).
        if self._build_queue is not None:
            # deferred: embeddings is a heavy import
            from rbtr.index import embeddings  # noqa: PLC0415

            embeddings.start_idle_monitor(config.embed_idle_timeout)
        self._setup_signal_handlers()

        ctx = zmq.asyncio.Context()
        rep: zmq.asyncio.Socket = ctx.socket(zmq.REP)
        pub: zmq.asyncio.Socket = ctx.socket(zmq.PUB)
        self._pub_socket = pub

        threads: list[threading.Thread] = []

        # Start build worker
        if self._build_queue is not None:
            threads.append(self._build_queue.start())

        # Start watcher
        watcher_thread = threading.Thread(target=self._watcher_loop, daemon=True)
        watcher_thread.start()
        threads.append(watcher_thread)

        try:
            rep.bind(self.rpc_addr)
            pub.bind(self.pub_addr)

            # Write status file only after sockets are bound so clients
            # never see a file with stale endpoints.
            write_status(
                self.sock_dir,
                pid=os.getpid(),
                rpc=self.rpc_addr,
                pub=self.pub_addr,
                version=get_version(),
            )
            log.info("Daemon listening: rpc=%s pub=%s", self.rpc_addr, self.pub_addr)

            while not self._shutdown:
                self._drain_notifications(pub)
                if await rep.poll(timeout=100):
                    raw = await rep.recv()
                    response = self._dispatch(raw)
                    await rep.send(response.model_dump_json().encode())
        finally:
            self._cleanup()
            self._pub_socket = None
            rep.close()
            pub.close()
            ctx.term()
            for t in threads:
                t.join(timeout=2)
            log.info("Daemon stopped")

    def _drain_notifications(self, pub: zmq.asyncio.Socket) -> None:
        while True:
            try:
                notification = self._notification_queue.get_nowait()
            except queue.Empty:
                break
            pub.send(notification.model_dump_json().encode(), zmq.NOBLOCK)

    def _next_poll_interval(self) -> float:
        """Pick the watcher poll interval based on build-queue state.

        While a build is active, slow the watcher down so a long
        embed phase doesn't flood the queue with duplicates for
        the same stale SHA. Other repos are still detected on the
        busy cadence — no repo is starved.
        """
        if self._build_queue is not None and self._build_queue.active_repo is not None:
            return self._busy_poll_interval
        return self._idle_poll_interval

    def _watcher_loop(self) -> None:
        while not self._shutdown:
            time.sleep(self._next_poll_interval())
            if self._shutdown:
                break
            if self._store is None:
                continue
            for stale in watcher.poll(self._store):
                log.info(
                    "HEAD not indexed in %s: %s",
                    stale.repo_path,
                    stale.new_ref[:12],
                )
                self._notification_queue.put(
                    AutoRebuildNotification(
                        repo=stale.repo_path,
                        new_ref=stale.new_ref,
                    )
                )
                if self._build_queue is not None:
                    self._build_queue.submit(stale.repo_path, [stale.new_ref])

    def _dispatch(self, raw: bytes) -> Response:
        try:
            request = request_adapter.validate_json(raw)
        except Exception as exc:
            return ErrorResponse(
                code=ErrorCode.INVALID_REQUEST,
                message=f"Invalid request: {exc}",
            )
        handler = self._handlers.get(request.kind)
        if handler is None:
            return ErrorResponse(
                code=ErrorCode.INVALID_REQUEST,
                message=f"No handler for kind: {request.kind}",
            )
        try:
            return handler(request)
        except Exception as exc:
            log.exception("Handler error for %s", request.kind)
            return ErrorResponse(code=ErrorCode.INTERNAL, message=str(exc))

    def _handle_shutdown(self, _request: ShutdownRequest) -> OkResponse:
        self.request_shutdown()
        return OkResponse()
