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
ref watcher. Handlers are registered by `kind` string and
receive the typed request model.

Lifecycle::

    server = DaemonServer(Path.home() / ".rbtr")
    anyio.run(server.serve)           # blocks until shutdown
    # or from another thread:
    server.request_shutdown()         # thread-safe
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import zmq
import zmq.asyncio

from rbtr import get_version
from rbtr.daemon.handlers import (
    handle_build_index_async,
    handle_changed_symbols,
    handle_find_refs,
    handle_list_symbols,
    handle_read_symbol,
    handle_search,
    handle_status,
)
from rbtr.daemon.messages import (
    ErrorCode,
    ErrorResponse,
    Notification,
    OkResponse,
    PingResponse,
    Request,
    Response,
    ShutdownRequest,
    request_adapter,
)
from rbtr.daemon.repos import RepoManager
from rbtr.daemon.watcher import RefWatcher
from rbtr.index.store import IndexStore

log = logging.getLogger(__name__)

type RequestHandler = Callable[[Any], Response]


class DaemonServer:
    """ZMQ daemon server managing REP and PUB sockets.

    Handlers are registered by request `kind` and receive the
    validated pydantic model. Built-in handlers: `ping`
    (returns version and uptime) and `shutdown` (sets the
    shutdown flag). Additional handlers are added via
    `register()`.
    """

    def __init__(
        self,
        sock_dir: Path,
        store: IndexStore | None = None,
        *,
        poll_interval: float = 5.0,
    ) -> None:
        self.sock_dir = sock_dir
        self.rpc_addr = f"ipc://{sock_dir / 'daemon.rpc'}"
        self.pub_addr = f"ipc://{sock_dir / 'daemon.pub'}"
        self._start_time = time.monotonic()
        self._shutdown = False
        self._pub_socket: zmq.asyncio.Socket | None = None
        self._notification_queue: queue.SimpleQueue[Notification] = queue.SimpleQueue()
        self._handlers: dict[str, RequestHandler] = {
            "ping": self._handle_ping,
            "shutdown": self._handle_shutdown,
        }
        self._mgr: RepoManager | None = None
        self._watcher = RefWatcher()
        self._poll_interval = poll_interval
        if store is not None:
            self._register_index_handlers(store)

    def _register_index_handlers(self, store: IndexStore) -> None:
        """Register all index method handlers."""
        mgr = RepoManager(store)
        self._mgr = mgr
        notify = self._notification_queue.put
        self._handlers.update(
            {
                "search": lambda req: handle_search(req, mgr),
                "read_symbol": lambda req: handle_read_symbol(req, mgr),
                "list_symbols": lambda req: handle_list_symbols(req, mgr),
                "find_refs": lambda req: handle_find_refs(req, mgr),
                "changed_symbols": lambda req: handle_changed_symbols(req, mgr),
                "status": lambda req: handle_status(req, mgr),
                "build_index": lambda req: handle_build_index_async(req, mgr, notify),
            }
        )

    def register(self, kind: str, handler: RequestHandler) -> None:
        """Register a handler for a request kind."""
        self._handlers[kind] = handler

    def request_shutdown(self) -> None:
        """Signal the server to stop (thread-safe)."""
        self._shutdown = True

    async def serve(self) -> None:
        """Run the REP + PUB event loop with watcher thread."""
        self.sock_dir.mkdir(parents=True, exist_ok=True)

        ctx = zmq.asyncio.Context()
        rep: zmq.asyncio.Socket = ctx.socket(zmq.REP)
        pub: zmq.asyncio.Socket = ctx.socket(zmq.PUB)
        self._pub_socket = pub

        # Start watcher thread if we have a store
        watcher_thread: threading.Thread | None = None
        if self._mgr is not None:
            watcher_thread = threading.Thread(target=self._watcher_loop, daemon=True)
            watcher_thread.start()

        try:
            rep.bind(self.rpc_addr)
            pub.bind(self.pub_addr)
            log.info("Daemon listening: rpc=%s pub=%s", self.rpc_addr, self.pub_addr)

            while not self._shutdown:
                # Drain pending notifications from build threads / watcher
                self._drain_notifications(pub)

                # Handle one RPC request if available
                if await rep.poll(timeout=100):
                    raw = await rep.recv()
                    response = self._dispatch(raw)
                    await rep.send(response.model_dump_json().encode())
        finally:
            self._pub_socket = None
            rep.close()
            pub.close()
            ctx.term()
            (self.sock_dir / "daemon.rpc").unlink(missing_ok=True)
            (self.sock_dir / "daemon.pub").unlink(missing_ok=True)
            if watcher_thread is not None:
                watcher_thread.join(timeout=2)
            log.info("Daemon stopped")

    def _drain_notifications(self, pub: zmq.asyncio.Socket) -> None:
        """Publish all queued notifications (non-blocking)."""
        while True:
            try:
                notification = self._notification_queue.get_nowait()
            except queue.Empty:
                break
            # ZMQ send from the event loop thread (sync send is
            # fine here since PUB never blocks).
            pub.send(notification.model_dump_json().encode(), zmq.NOBLOCK)

    def _watcher_loop(self) -> None:
        """Poll registered repos for HEAD changes (runs in thread)."""
        while not self._shutdown:
            time.sleep(self._poll_interval)
            if self._shutdown:
                break
            changes = self._watcher.poll()
            for change in changes:
                log.info(
                    "HEAD changed in %s: %s → %s",
                    change.repo_path,
                    change.old_ref[:12],
                    change.new_ref[:12],
                )
                from rbtr.daemon.messages import AutoRebuildNotification

                self._notification_queue.put(
                    AutoRebuildNotification(
                        repo=change.repo_path,
                        old_ref=change.old_ref,
                        new_ref=change.new_ref,
                    )
                )
                # Trigger build for the new ref
                if self._mgr is not None:
                    from rbtr.daemon.handlers import _do_build

                    _do_build(
                        change.repo_path,
                        [change.new_ref],
                        self._mgr,
                        self._notification_queue.put,
                    )

    async def publish(self, notification: Notification) -> None:
        """Broadcast a notification to all SUB clients."""
        if self._pub_socket is not None:
            await self._pub_socket.send(notification.model_dump_json().encode())

    def _dispatch(self, raw: bytes) -> Response:
        """Parse a request and route to the matching handler."""
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
            return ErrorResponse(
                code=ErrorCode.INTERNAL,
                message=str(exc),
            )

    def _handle_ping(self, _request: Request) -> PingResponse:
        return PingResponse(
            version=get_version(),
            uptime=round(time.monotonic() - self._start_time, 1),
        )

    def _handle_shutdown(self, _request: ShutdownRequest) -> OkResponse:
        self.request_shutdown()
        return OkResponse()
