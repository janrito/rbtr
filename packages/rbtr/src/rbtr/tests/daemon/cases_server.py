"""Cases for how a serving daemon stops.

Each case builds a daemon, says what makes it stop, and says what
stopping shows: the errors `serve()` raises and the errors it logs.
"""

from __future__ import annotations

import asyncio
import errno
import threading
from collections.abc import Callable, Coroutine, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pygit2
import pytest
import structlog
import zmq
import zmq.asyncio
from pytest_cases import parametrize
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    DaemonConfigRequest,
    ErrorCode,
    ErrorResponse,
    OkResponse,
    Request,
    Response,
    SearchRequest,
    ShutdownRequest,
    WatchRequest,
    response_adapter,
)
from rbtr.daemon.server import Daemon
from rbtr.daemon.status import remove_status
from rbtr.domain.models import Chunk
from rbtr.git import FileEntry, normalise_repo_path
from rbtr.index import build, maintenance
from rbtr.index.index import Index
from rbtr.index.indexer import Indexer
from rbtr.index.progress import RepoProgressCallback
from rbtr.tests.conftest import StubModel


@dataclass(frozen=True)
class StopScenario:
    """A daemon, what makes it stop once it serves, and what stopping shows.

    `raised` holds the messages of the errors `serve()` raises, and
    `logged` the events it logs at error level.
    """

    server: Daemon
    stop: Callable[[], None]
    raised: list[str]
    logged: list[str]


@dataclass(frozen=True)
class MalformedMessage:
    client: zmq.Socket[bytes]
    frames: list[bytes]
    reply_prefix: list[bytes]


def message_invalid_json(request_peer: zmq.Socket[bytes]) -> MalformedMessage:
    return MalformedMessage(request_peer, [b"not json"], [])


def message_correlated_invalid_json(correlated_request_peer: zmq.Socket[bytes]) -> MalformedMessage:
    return MalformedMessage(correlated_request_peer, [b"not json"], [])


def message_multiple_bodies(rpc_peer: zmq.Socket[bytes]) -> MalformedMessage:
    return MalformedMessage(
        rpc_peer, [b"return-hop", b"", b"{}", b"extra-body"], [b"return-hop", b""]
    )


def case_shutdown_request(runtime_dir: Path) -> StopScenario:
    """A shutdown request is answered, and the daemon stops cleanly."""
    server = Daemon(runtime_dir, index=None, idle_poll_interval=60.0, busy_poll_interval=60.0)

    def stop() -> None:
        with DaemonClient(runtime_dir) as client:
            assert isinstance(client.send(ShutdownRequest()), OkResponse)

    return StopScenario(server=server, stop=stop, raised=[], logged=[])


@parametrize(model_name=["embeddings", "reranker"])
@parametrize(fails=[False, True])
@parametrize(companion_fails=[False, True])
def case_shutdown_during_warmup(
    model_name: str,
    fails: bool,
    companion_fails: bool,
    runtime_dir: Path,
    empty_index: Index,
    stub_embedding_model: None,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    log_output: structlog.testing.LogCapture,
) -> Generator[StopScenario]:
    """Scheduled model loads finish and dispose even on fatal stop."""
    warmup_model = StubModel()

    def dispose() -> None:
        assert not any(entry["event"] == "daemon_stopped" for entry in log_output.entries)

    disposed = mocker.patch.object(warmup_model, "close", side_effect=dispose)
    monkeypatch.setattr(config, "warmup", True)
    monkeypatch.setattr(config, "reranker_model", "stub-reranker")
    started = threading.Event()
    release = threading.Event()
    fail = threading.Event()

    def load_model() -> StubModel:
        started.set()
        assert release.wait(timeout=10), "warmup loader was not released"
        return warmup_model

    if companion_fails:
        companion = "reranker" if model_name == "embeddings" else "embeddings"
        mocker.patch(f"rbtr.index.{companion}._load_model", side_effect=RuntimeError("warmup"))
    if model_name == "embeddings":
        mocker.patch("rbtr.index.embeddings._load_model", side_effect=lambda _abort: load_model())
    else:
        mocker.patch("rbtr.index.reranker._load_model", side_effect=load_model)
    server = Daemon(runtime_dir, index=empty_index, idle_poll_interval=60.0)

    def interval() -> float:
        if fail.is_set():
            msg = "unexpected"
            raise RuntimeError(msg)
        return 0.01

    if fails:
        mocker.patch.object(Indexer, "poll_interval", side_effect=interval)

    def stop() -> None:
        try:
            assert started.wait(timeout=5), "warmup task did not start"
            if fails:
                fail.set()
                assert server._stopping.wait(timeout=5), "fatal stop did not begin"
            else:
                with DaemonClient(runtime_dir) as client:
                    assert isinstance(client.send(ShutdownRequest()), OkResponse)
        finally:
            release.set()

    yield StopScenario(
        server=server,
        stop=stop,
        raised=["unexpected"] if fails else [],
        logged=["task_failed"] if fails else [],
    )
    disposed.assert_called_once()


def case_shutdown_during_build(
    runtime_dir: Path,
    git_repo: pygit2.Repository,
    index: Index,
    write_started: threading.Event,
    write_released: threading.Event,
    mocker: MockerFixture,
) -> StopScenario:
    """A shutdown during a build ends it after the file in hand, as a stop
    rather than a failure: no error is published or logged."""
    repo_path = normalise_repo_path(git_repo.workdir)
    index.register(repo_path)
    server = Daemon(runtime_dir, index=index, idle_poll_interval=60.0, busy_poll_interval=60.0)
    extract = build.extract_file

    def held_extract(entry: FileEntry, language: str) -> list[Chunk]:
        write_started.set()
        assert write_released.wait(20), "file extraction was not released"
        return extract(entry, language)

    mocker.patch.object(build, "extract_file", side_effect=held_extract)

    def stop() -> None:
        try:
            assert write_started.wait(10), "initial build did not start"
            server.request_shutdown()
        finally:
            write_released.set()

    return StopScenario(server=server, stop=stop, raised=[], logged=[])


def case_shutdown_during_maintenance(
    runtime_dir: Path,
    empty_index: Index,
    write_started: threading.Event,
    write_released: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> Generator[StopScenario]:
    """Native maintenance and its output finish before transport disposal."""
    monkeypatch.setattr(config, "embed", False)
    monkeypatch.setattr(config, "reranker_model", None)
    monkeypatch.setattr(config, "maintenance_idle_s", 0)
    server = Daemon(runtime_dir, empty_index, idle_poll_interval=0.01)
    removed = mocker.patch("rbtr.daemon.server.remove_status", wraps=remove_status)
    cleanup_seen: list[bool] = []
    loop_seen: Future[asyncio.AbstractEventLoop] = Future()
    submit = asyncio.run_coroutine_threadsafe

    def capture[Result](
        coroutine: Coroutine[None, None, Result], loop: asyncio.AbstractEventLoop
    ) -> Future[Result]:
        if not loop_seen.done():
            loop_seen.set_result(loop)
        return submit(coroutine, loop)

    def work(index: Index, *, now: datetime, on_progress: RepoProgressCallback) -> None:
        write_started.set()
        assert write_released.wait(20), "maintenance was not released"
        maintenance.run_maintenance(index, now=now, on_progress=on_progress)
        cleanup_seen.append(removed.called)

    async def advance() -> None:
        # Give shutdown continuations loop turns while the native pass is held.
        for _ in range(4):
            await asyncio.sleep(0)

    mocker.patch.object(asyncio, "run_coroutine_threadsafe", side_effect=capture)
    mocker.patch("rbtr.index.indexer.run_maintenance", side_effect=work)

    def stop() -> None:
        try:
            assert write_started.wait(10), "maintenance did not start"
            with DaemonClient(runtime_dir) as client:
                assert isinstance(client.send(ShutdownRequest()), OkResponse)
            submit(advance(), loop_seen.result(timeout=10)).result(timeout=10)
        finally:
            write_released.set()

    yield StopScenario(server=server, stop=stop, raised=[], logged=[])
    assert cleanup_seen == [False], "transport was disposed before maintenance finished"


def case_job_worker_fails(
    runtime_dir: Path, empty_index: Index, fake_repo: str, mocker: MockerFixture
) -> StopScenario:
    """An error the job worker does not expect stops the daemon, logged,
    rather than leaving it answering reads with nothing being built."""
    server = Daemon(
        runtime_dir, index=empty_index, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    mocker.patch.object(Indexer, "next_job", side_effect=RuntimeError("unexpected"))

    def stop() -> None:
        # A watch wakes the worker, which looks for its next job.
        with DaemonClient(runtime_dir) as client:
            client.send(WatchRequest(repo_path=fake_repo, refs=["HEAD"]))

    return StopScenario(server=server, stop=stop, raised=["unexpected"], logged=["task_failed"])


def case_rejected_watch_does_not_wake_worker(
    runtime_dir: Path, empty_index: Index, mocker: MockerFixture
) -> StopScenario:
    server = Daemon(runtime_dir, empty_index, idle_poll_interval=60.0)
    initial_selection = threading.Event()
    select = Indexer.next_job

    def next_job(service: Indexer) -> None:
        if initial_selection.is_set():
            raise RuntimeError("unexpected")
        assert select(service) is None
        initial_selection.set()

    mocker.patch.object(Indexer, "next_job", next_job)

    def stop() -> None:
        assert initial_selection.wait(10), "initial empty selection did not finish"
        with DaemonClient(runtime_dir) as client:
            response = client.send(WatchRequest(repo_path=str(runtime_dir), refs=["HEAD"]))
            assert isinstance(response, ErrorResponse)
            assert response.code is ErrorCode.REPO_NOT_FOUND
            assert isinstance(client.send(ShutdownRequest()), OkResponse)

    return StopScenario(server=server, stop=stop, raised=[], logged=[])


def case_watcher_fails(
    runtime_dir: Path, empty_index: Index, mocker: MockerFixture
) -> StopScenario:
    """An error the watcher does not expect stops the daemon, logged,
    rather than leaving it answering reads with nothing being watched."""
    server = Daemon(
        runtime_dir, index=empty_index, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    mocker.patch.object(Indexer, "poll_interval", side_effect=RuntimeError("unexpected"))
    return StopScenario(
        server=server, stop=lambda: None, raised=["unexpected"], logged=["task_failed"]
    )


@parametrize(fails=[False, True])
def case_stop_with_running_and_queued_rpc(
    fails: bool,
    runtime_dir: Path,
    seeded_index: Index,
    fake_repo: str,
    held_query: None,
    query_started: threading.Event,
    query_released: threading.Event,
    rpc_send: Callable[[Request], Response],
    rpc_peer: zmq.Socket[bytes],
    mocker: MockerFixture,
) -> StopScenario:
    server = Daemon(runtime_dir, seeded_index, idle_poll_interval=60)
    fail = threading.Event()

    def interval() -> float:
        if fail.is_set():
            msg = "unexpected"
            raise RuntimeError(msg)
        return 0.01

    if fails:
        mocker.patch.object(Indexer, "poll_interval", side_effect=interval)

    def stop() -> None:
        with ThreadPoolExecutor(max_workers=1) as clients:
            active = clients.submit(
                rpc_send, SearchRequest(repo_path=fake_repo, query="load_config")
            )
            try:
                assert query_started.wait(5)
                rpc_peer.send_multipart(
                    [
                        b"queued",
                        b"",
                        SearchRequest(repo_path=fake_repo, query="Application")
                        .model_dump_json()
                        .encode(),
                    ]
                )
                rpc_peer.send_multipart(
                    [b"fence", b"", DaemonConfigRequest().model_dump_json().encode()]
                )
                frames = rpc_peer.recv_multipart()
                assert frames[:2] == [b"fence", b""]
                assert response_adapter.validate_json(frames[-1]).kind == "daemon_config"
                if fails:
                    fail.set()
                else:
                    server.request_shutdown()
                frames = rpc_peer.recv_multipart()
                assert frames[:2] == [b"queued", b""]
                response = response_adapter.validate_json(frames[-1])
                assert isinstance(response, ErrorResponse)
                assert response.code is ErrorCode.SHUTTING_DOWN
                assert not active.done()
            finally:
                query_released.set()
            assert active.result(timeout=5).kind == "search"

    return StopScenario(
        server=server,
        stop=stop,
        raised=["unexpected"] if fails else [],
        logged=["task_failed"] if fails else [],
    )


def case_reply_transport_fails(runtime_dir: Path, mocker: MockerFixture) -> StopScenario:
    server = Daemon(runtime_dir, idle_poll_interval=60)
    error = zmq.ZMQError(errno.EIO)
    mocker.patch.object(zmq.asyncio.Socket, "send_multipart", side_effect=error)

    def stop() -> None:
        with zmq.Context() as context, context.socket(zmq.REQ) as client:
            client.setsockopt(zmq.LINGER, 0)
            client.connect(server.rpc_addr)
            client.send(DaemonConfigRequest().model_dump_json().encode())
            assert client.poll(2000) == 0

    return StopScenario(server=server, stop=stop, raised=[str(error)], logged=["task_failed"])
