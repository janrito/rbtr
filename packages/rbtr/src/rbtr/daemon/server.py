"""ZMQ daemon server.

Binds two sockets in `sock_dir`:

- **REP** (`daemon.rpc`) — synchronous request/response.
  Receives a `Request` (discriminated on `kind`), dispatches
  to the matching handler, returns a `Response`.
- **PUB** (`daemon.pub`) — fan-out notifications.
  Broadcasts `Notification` messages (index progress,
  ready, auto-rebuild) to any connected SUB clients (pi-rbtr
  extension, CLI listeners).

The event loop polls REP with a 100 ms timeout so the
shutdown flag is checked promptly. Handlers are registered
by `kind` string and receive the typed request model.

Lifecycle::

    server = DaemonServer(Path.home() / ".rbtr")
    anyio.run(server.serve)           # blocks until shutdown
    # or from another thread:
    server.request_shutdown()         # thread-safe
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import zmq
import zmq.asyncio

from rbtr import get_version
from rbtr.daemon.handlers import (
    handle_build_index,
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

    def __init__(self, sock_dir: Path, store: IndexStore | None = None) -> None:
        self.sock_dir = sock_dir
        self.rpc_addr = f"ipc://{sock_dir / 'daemon.rpc'}"
        self.pub_addr = f"ipc://{sock_dir / 'daemon.pub'}"
        self._start_time = time.monotonic()
        self._shutdown = False
        self._pub_socket: zmq.asyncio.Socket | None = None
        self._handlers: dict[str, RequestHandler] = {
            "ping": self._handle_ping,
            "shutdown": self._handle_shutdown,
        }
        if store is not None:
            self._register_index_handlers(store)

    def _register_index_handlers(self, store: IndexStore) -> None:
        """Register all index method handlers."""
        mgr = RepoManager(store)
        self._handlers.update(
            {
                "search": lambda req: handle_search(req, mgr),
                "read_symbol": lambda req: handle_read_symbol(req, mgr),
                "list_symbols": lambda req: handle_list_symbols(req, mgr),
                "find_refs": lambda req: handle_find_refs(req, mgr),
                "changed_symbols": lambda req: handle_changed_symbols(req, mgr),
                "status": lambda req: handle_status(req, mgr),
                "build_index": lambda req: handle_build_index(req, mgr),
            }
        )

    def register(self, kind: str, handler: RequestHandler) -> None:
        """Register a handler for a request kind."""
        self._handlers[kind] = handler

    def request_shutdown(self) -> None:
        """Signal the server to stop (thread-safe)."""
        self._shutdown = True

    async def serve(self) -> None:
        """Run the REP + PUB event loop."""
        self.sock_dir.mkdir(parents=True, exist_ok=True)

        ctx = zmq.asyncio.Context()
        rep: zmq.asyncio.Socket = ctx.socket(zmq.REP)
        pub: zmq.asyncio.Socket = ctx.socket(zmq.PUB)
        self._pub_socket = pub

        try:
            rep.bind(self.rpc_addr)
            pub.bind(self.pub_addr)
            log.info("Daemon listening: rpc=%s pub=%s", self.rpc_addr, self.pub_addr)

            while not self._shutdown:
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
            log.info("Daemon stopped")

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
