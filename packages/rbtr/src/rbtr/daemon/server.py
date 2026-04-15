"""ZMQ daemon server.

Binds two sockets in `sock_dir`:

- **REP** (`daemon.rpc`) — synchronous request/response.
  Receives a `Request` (discriminated on `kind`), dispatches
  to the matching handler, returns a `Response`.
- **PUB** (`daemon.pub`) — fan-out notifications.
  Broadcasts `Notification` messages (build progress, index
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
from pathlib import Path
from typing import Any

import zmq
import zmq.asyncio

from rbtr.daemon.messages import (
    ErrorCode,
    ErrorResponse,
    Notification,
    PingRequest,
    PingResponse,
    Response,
    ShutdownRequest,
    ShutdownResponse,
    request_adapter,
)

log = logging.getLogger(__name__)

type Handler = dict[str, Any]


class DaemonServer:
    """ZMQ daemon server managing REP and PUB sockets.

    Handlers are registered by request `kind` and receive the
    validated pydantic model. Built-in handlers: `ping`
    (returns version and uptime) and `shutdown` (sets the
    shutdown flag). Additional handlers are added via
    `register()`.
    """

    def __init__(self, sock_dir: Path) -> None:
        self.sock_dir = sock_dir
        self.rpc_addr = f"ipc://{sock_dir / 'daemon.rpc'}"
        self.pub_addr = f"ipc://{sock_dir / 'daemon.pub'}"
        self._start_time = time.monotonic()
        self._shutdown = False
        self._pub_socket: zmq.asyncio.Socket | None = None
        self._handlers: dict[str, Any] = {
            "ping": self._handle_ping,
            "shutdown": self._handle_shutdown,
        }

    def register(self, kind: str, handler: Any) -> None:
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

        rep.bind(self.rpc_addr)
        pub.bind(self.pub_addr)
        log.info("Daemon listening: rpc=%s pub=%s", self.rpc_addr, self.pub_addr)

        try:
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

    def _handle_ping(self, request: PingRequest) -> PingResponse:
        return PingResponse(
            version="0.1.0",
            uptime=round(time.monotonic() - self._start_time, 1),
        )

    def _handle_shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        self.request_shutdown()
        return ShutdownResponse()
