"""Receive client requests, execute supplied handlers and return their replies."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from uuid import uuid4

import duckdb
import structlog
import zmq
import zmq.asyncio
from pydantic import ValidationError

from rbtr.config import config
from rbtr.daemon.messages import (
    ErrorCode,
    ErrorResponse,
    HasRepoPath,
    OkResponse,
    Request,
    Response,
    ShutdownRequest,
    request_adapter,
)
from rbtr.errors import IndexNotBuiltError, RbtrError
from rbtr.git import normalise_repo_path
from rbtr.logging import elapsed_ms

log = structlog.get_logger(__name__)

# A validated kind selects a concrete request model from heterogeneous handlers.
type RequestHandler = Callable[[Any], Response]


def _ignore_result(_response: Response) -> None:
    pass


class RpcResource(Enum):
    """Select the request queue and the shared resource held during execution.

    READ needs neither shared lock; GPU takes the supplied GPU lock; WRITE takes
    the supplied writer semaphore. The class applies to the whole handler call.
    """

    READ = auto()
    GPU = auto()
    WRITE = auto()


class RpcPhase(Enum):
    """Shutdown eligibility: QUEUED may be rejected; RUNNING must finish.

    Executor submissions remain QUEUED until a native thread claims execution.
    REJECTED identifies calls already claimed by shutdown.
    """

    QUEUED = auto()
    RUNNING = auto()
    REJECTED = auto()


@dataclass(frozen=True)
class Handler:
    """Register a synchronous operation, its resource class and result callback.

    `call` receives the validated request model for its registration key and
    returns a response on an executor thread. `on_result` is a short synchronous
    callback on the serving loop, after resource release and before the reply.
    It receives returned responses, including errors returned by `call`; raised
    exceptions use RPC error handling instead.
    """

    resource: RpcResource
    call: RequestHandler
    on_result: Callable[[Response], None] = _ignore_result


@dataclass(eq=False)
class RpcCall:
    """One accepted request retained until its reply or rejection is handled.

    `envelope` is the complete ZeroMQ return address through the empty delimiter;
    it must be reused unchanged. `phase` is updated under RPC's native-start
    lock. Calls have distinct identities even when their requests are equal.
    """

    request: Request
    handler: Handler
    envelope: list[bytes]
    phase: RpcPhase = RpcPhase.QUEUED


@dataclass(frozen=True)
class RpcQueue:
    """A loop-owned request FIFO with a worker count and execution resource.

    `capacity` limits async workers; pending requests are uncapped.
    A `None` item tells one worker to exit.
    """

    capacity: int
    resource: contextlib.AbstractAsyncContextManager[None]
    pending: asyncio.Queue[RpcCall | None] = field(default_factory=asyncio.Queue)


def _format_validation_error(exc: ValidationError) -> str:
    """Return field paths, validation messages and shortened received values."""
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        received = repr(err.get("input"))
        if len(received) > 120:
            received = f"{received[:117]}..."
        lines.append(f"  - {loc}: {err['msg']} (received: {received})")
    body = "\n".join(lines) or "  - (no detail)"
    return f"Invalid request arguments:\n{body}"


class RPC:
    """Answer clients through supplied synchronous request handlers.

    Use once on one event loop. The context and GPU/write controls remain
    caller-owned; RPC owns its request socket. After `request_shutdown`, await
    `serve` before calling `close` on that loop and disposing of the context.
    """

    def __init__(
        self,
        context: zmq.asyncio.Context,
        endpoint: str,
        *,
        handlers: Mapping[str, Handler],
        gpu_lock: asyncio.Lock,
        write_sem: asyncio.Semaphore,
        index_error: Callable[[IndexNotBuiltError], ErrorResponse],
        on_shutdown: Callable[[], None],
        on_failure: Callable[[asyncio.Task[None]], None],
    ) -> None:
        """Configure handlers, shared resources and serving callbacks.

        `handlers` maps request kinds to matching handlers. Supply the same
        GPU/write locks used by other work sharing those resources.

        Callbacks have these contracts:

        - `index_error(error)` returns an `ErrorResponse` for a missing index.
          It runs on an executor thread after the handler releases its resource.
        - `on_shutdown()` requests application shutdown on the serving loop.
          It can be called more than once and must tolerate repeated calls.
        - `on_failure(task)` runs on that loop when a queue worker finishes.
          The caller inspects the task to distinguish failure from normal exit.

        Loop callbacks must return promptly.
        """
        self._context = context
        self._endpoint = endpoint
        self._handlers = dict(handlers)
        self._index_error = index_error
        self._on_shutdown = on_shutdown
        self._on_failure = on_failure
        self._shutdown = False
        self._rpc_lock = threading.Lock()
        self._rpc_calls: set[RpcCall] = set()
        self._rpc_waiting: set[asyncio.Task[None]] = set()
        self._router: zmq.asyncio.Socket | None = None
        self._rpc_queues = {
            RpcResource.READ: RpcQueue(config.max_concurrent_reads, contextlib.nullcontext()),
            RpcResource.GPU: RpcQueue(1, gpu_lock),
            RpcResource.WRITE: RpcQueue(1, write_sem),
        }

    def request_shutdown(self) -> None:
        """Request stop without waiting; safe to call repeatedly from any thread."""
        with self._rpc_lock:
            self._shutdown = True

    def close(self) -> None:
        """Close the request socket on its loop after `serve`; repeated calls are safe."""
        if self._router is not None:
            self._router.close()
            self._router = None

    async def serve(self, on_ready: Callable[[], None]) -> None:
        """Bind and answer requests until stop; leave the socket open for `close`.

        `on_ready()` runs on the serving loop after binding, before intake.
        On stop, reject not-started calls and join started handlers and reply
        attempts before acknowledging shutdown. Handler errors become responses;
        transport and worker failures end serving with an exception.
        """
        # Contexts are shareable; PyZMQ sockets are created, used and closed
        # on their owning loop. The caller retains the socket through teardown.
        router: zmq.asyncio.Socket = self._context.socket(zmq.ROUTER)
        self._router = router
        router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        router.setsockopt(zmq.LINGER, 0)
        router.bind(self._endpoint)
        on_ready()
        workers = [
            asyncio.create_task(
                self._rpc_worker(router, resource), name=f"rpc_{resource.name.lower()}_{i}"
            )
            for resource, queue in self._rpc_queues.items()
            for i in range(queue.capacity)
        ]
        for task in workers:
            task.add_done_callback(self._on_failure)
        shutdown_envelope: list[bytes] | None = None
        try:
            while not self._shutdown:
                if not await router.poll(timeout=100):
                    continue
                frames = await router.recv_multipart()
                try:
                    delimiter = frames.index(b"")
                except ValueError:
                    log.warning("rpc_invalid_envelope")
                    continue
                envelope = frames[: delimiter + 1]
                if delimiter == 0:
                    log.warning("rpc_invalid_envelope")
                    continue
                if len(frames) != delimiter + 2:
                    await self._reply(
                        router,
                        envelope,
                        ErrorResponse(
                            code=ErrorCode.INVALID_REQUEST, message="Expected one request body."
                        ),
                    )
                    continue
                try:
                    request = request_adapter.validate_json(frames[-1])
                except ValidationError as exc:
                    await self._reply(
                        router,
                        envelope,
                        ErrorResponse(
                            code=ErrorCode.INVALID_REQUEST,
                            message=_format_validation_error(exc),
                        ),
                    )
                    continue
                if isinstance(request, ShutdownRequest):
                    shutdown_envelope = envelope
                    self.request_shutdown()
                    self._on_shutdown()
                    break
                handler = self._handlers.get(request.kind)
                if handler is None:
                    await self._reply(
                        router,
                        envelope,
                        ErrorResponse(
                            code=ErrorCode.INVALID_REQUEST,
                            message=f"No handler for kind: {request.kind}",
                        ),
                    )
                    continue
                call = RpcCall(request, handler, envelope)
                with self._rpc_lock:
                    self._rpc_calls.add(call)
                self._rpc_queues[handler.resource].pending.put_nowait(call)
        finally:
            self.request_shutdown()
            self._on_shutdown()
            await self._finish_rpcs(router, workers)
            if shutdown_envelope is not None:
                await self._reply(router, shutdown_envelope, OkResponse())

    async def _reply(
        self, router: zmq.asyncio.Socket, envelope: list[bytes], response: Response
    ) -> None:
        try:
            # PyZMQ sockets stay on their owning event loop. A missing or
            # non-reading peer must not hold a worker or shutdown open.
            await router.send_multipart(
                [*envelope, response.model_dump_json().encode()], flags=zmq.DONTWAIT
            )
        except zmq.ZMQError as exc:
            if exc.errno not in (zmq.EHOSTUNREACH, zmq.EAGAIN):
                raise
            log.warning("rpc_reply_unavailable", errno=exc.errno)

    async def _rpc_worker(self, router: zmq.asyncio.Socket, resource: RpcResource) -> None:
        """Consume one request class until its stop marker, completing reply attempts."""
        queue = self._rpc_queues[resource].pending
        try:
            while True:
                call = await queue.get()
                try:
                    if call is None:
                        return
                    response = await self._dispatch(call)
                    if call.phase is not RpcPhase.REJECTED:
                        if self._shutdown and call.phase is not RpcPhase.RUNNING:
                            response = ErrorResponse(
                                code=ErrorCode.SHUTTING_DOWN, message="Daemon is shutting down."
                            )
                        await self._reply(router, call.envelope, response)
                finally:
                    queue.task_done()
                    if call is not None:
                        with self._rpc_lock:
                            self._rpc_calls.discard(call)
        except Exception:
            self.request_shutdown()
            raise

    async def _finish_rpcs(
        self, router: zmq.asyncio.Socket, workers: list[asyncio.Task[None]]
    ) -> None:
        """Reject unstarted calls and join every worker before returning or raising.

        Requires shutdown to have been requested. Submitted native functions
        finish their start check or execution; only resource waits are cancelled.
        """
        with self._rpc_lock:
            queued = [call for call in self._rpc_calls if call.phase is not RpcPhase.RUNNING]
            for call in queued:
                call.phase = RpcPhase.REJECTED
        try:
            for call in queued:
                await self._reply(
                    router,
                    call.envelope,
                    ErrorResponse(code=ErrorCode.SHUTTING_DOWN, message="Daemon is shutting down."),
                )
        finally:
            # Only tasks still acquiring a resource are cancellable. Tasks
            # awaiting an executor job must finish its native start check.
            for task in self._rpc_waiting:
                task.cancel()
            for rpc_queue in self._rpc_queues.values():
                queue = rpc_queue.pending
                while not queue.empty():
                    queue.get_nowait()
                    queue.task_done()
                for _ in range(rpc_queue.capacity):
                    queue.put_nowait(None)
            results = await asyncio.gather(*workers, return_exceptions=True)
            for rpc_queue in self._rpc_queues.values():
                queue = rpc_queue.pending
                while not queue.empty():
                    queue.get_nowait()
                    queue.task_done()
            with self._rpc_lock:
                self._rpc_calls.difference_update(queued)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise result

    def _invoke(self, call: RpcCall) -> Response:
        """Invoke on an executor thread with the call's execution resource held."""
        # Claim native execution atomically with request_shutdown. An
        # executor submission is still queued until this check succeeds.
        with self._rpc_lock:
            if self._shutdown or call.phase is RpcPhase.REJECTED:
                return ErrorResponse(
                    code=ErrorCode.SHUTTING_DOWN, message="Daemon is shutting down."
                )
            call.phase = RpcPhase.RUNNING
        request = call.request
        if isinstance(request, HasRepoPath) and request.repo_path is not None:
            try:
                request.repo_path = normalise_repo_path(request.repo_path)
            except RbtrError as exc:
                # A bad repo_path (e.g. a client whose cwd is not a git
                # repo) must not take the whole daemon down: reply with an
                # error so the surviving daemon keeps serving other repos.
                log.warning("request_rejected", kind=request.kind, error=str(exc))
                return ErrorResponse(code=ErrorCode.REPO_NOT_FOUND, message=str(exc))
            structlog.contextvars.bind_contextvars(repo=request.repo_path)
        return call.handler.call(request)

    async def _dispatch(self, call: RpcCall) -> Response:
        """Run a call under its resource, mapping ordinary handler failures to responses."""
        request = call.request
        # Bind correlation context for this request. All keys reset on
        # exit, so no state leaks to the next request.
        req_ctx: dict[str, str] = {"request_id": uuid4().hex[:8], "kind": request.kind}
        if isinstance(request, HasRepoPath):
            req_ctx["repo"] = request.repo_path
        task = asyncio.current_task()
        if task is None:
            msg = "RPC dispatch requires an asyncio task"
            raise RuntimeError(msg)
        with structlog.contextvars.bound_contextvars(**req_ctx):
            t0 = time.perf_counter()
            try:
                self._rpc_waiting.add(task)
                try:
                    async with self._rpc_queues[call.handler.resource].resource:
                        self._rpc_waiting.discard(task)
                        with self._rpc_lock:
                            if self._shutdown:
                                return ErrorResponse(
                                    code=ErrorCode.SHUTTING_DOWN, message="Daemon is shutting down."
                                )
                        result = await asyncio.to_thread(self._invoke, call)
                finally:
                    self._rpc_waiting.discard(task)
                    # to_thread copies context; carry the canonical path back
                    # into this request's loop-side completion/error logs.
                    if isinstance(request, HasRepoPath):
                        structlog.contextvars.bind_contextvars(repo=request.repo_path)
                call.handler.on_result(result)
            except IndexNotBuiltError as exc:
                # Build-awareness may read data; preserve its position after
                # resource release and keep the work off this loop.
                response = await asyncio.to_thread(self._index_error, exc)
                if response.code is not ErrorCode.INDEX_IN_PROGRESS:
                    log.warning("handler_error", error=str(exc), elapsed_ms=elapsed_ms(t0))
                return response
            except RbtrError as exc:
                log.warning("handler_error", error=str(exc), elapsed_ms=elapsed_ms(t0))
                return ErrorResponse(code=ErrorCode(exc.error_code), message=str(exc))
            except duckdb.OutOfMemoryException as exc:
                log.warning("handler_error", error=str(exc), elapsed_ms=elapsed_ms(t0))
                return ErrorResponse(code=ErrorCode.OUT_OF_MEMORY, message=str(exc))
            except Exception as exc:
                log.exception("handler_error", elapsed_ms=elapsed_ms(t0))
                return ErrorResponse(code=ErrorCode.INTERNAL, message=str(exc))
            log.info("request_complete", elapsed_ms=elapsed_ms(t0))
            return result
