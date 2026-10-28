"""Real RPC completion order and resource-independent execution."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest
import structlog
import zmq
from pytest_cases import parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.daemon import rpc as daemon_rpc
from rbtr.daemon.messages import (
    DaemonConfigRequest,
    ErrorCode,
    ErrorResponse,
    ForgetRequest,
    GcRequest,
    Request,
    Response,
    SearchRequest,
    SearchResponse,
    ShutdownRequest,
    StatusRequest,
    WatchRequest,
    response_adapter,
)
from rbtr.daemon.server import Daemon
from rbtr.index.index import Index

from .conftest import serving


@pytest.fixture
def index(isolated_db: Path) -> Generator[Index]:
    """Use real project files for the concurrent switch/forget cases."""
    store = Index.from_config(writable=True)
    yield store
    store.close()


def test_config_replies_while_repository_discovery_is_running(
    running_daemon: Daemon,
    fake_repo: str,
    rpc_send: Callable[[Request], Response],
    mocker: MockerFixture,
) -> None:
    started = threading.Event()
    released = threading.Event()
    normalise = daemon_rpc.normalise_repo_path

    def normalise_repo_path(path: str) -> str:
        started.set()
        assert released.wait(20), "repository discovery was not released"
        return normalise(path)

    mocker.patch.object(daemon_rpc, "normalise_repo_path", side_effect=normalise_repo_path)
    with ThreadPoolExecutor(max_workers=2) as clients:
        scoped = clients.submit(rpc_send, StatusRequest(repo_path=fake_repo))
        try:
            assert started.wait(5)
            configured = clients.submit(rpc_send, DaemonConfigRequest()).result(timeout=5)
            assert configured.kind == "daemon_config"
            assert not scoped.done()
        finally:
            released.set()
        assert scoped.result(timeout=5).kind == "status"


@parametrize_with_cases("read_request", cases=".cases_rpc")
def test_reads_finish_while_gpu_and_writer_requests_wait(
    read_request: Request,
    held_query: None,
    held_watch: None,
    running_daemon: Daemon,
    fake_repo: str,
    query_started: threading.Event,
    query_released: threading.Event,
    write_started: threading.Event,
    write_released: threading.Event,
    rpc_send: Callable[[Request], Response],
    rpc_peer: zmq.Socket[bytes],
    log_output: structlog.testing.LogCapture,
) -> None:
    with ThreadPoolExecutor(max_workers=2) as clients:
        search = clients.submit(rpc_send, SearchRequest(repo_path=fake_repo, query="load_config"))
        writer = clients.submit(rpc_send, WatchRequest(repo_path=fake_repo, refs=["main"]))
        try:
            assert query_started.wait(5)
            assert write_started.wait(5)
            requests: list[tuple[bytes, Request]] = [
                (b"search", SearchRequest(repo_path=fake_repo, query="Application")),
                (b"watch", WatchRequest(repo_path=fake_repo, refs=["HEAD"])),
                (b"read", read_request),
            ]
            for tag, request in requests:
                rpc_peer.send_multipart([tag, b"", request.model_dump_json().encode()])
            frames = rpc_peer.recv_multipart()
            assert frames[:2] == [b"read", b""]
            response = response_adapter.validate_json(frames[-1])
            assert response.kind == read_request.kind
            assert not search.done()
            assert not writer.done()
        finally:
            query_released.set()
            write_released.set()
        assert isinstance(search.result(timeout=5), SearchResponse)
        assert writer.result(timeout=5).kind == "watch_set"
        replies = [rpc_peer.recv_multipart() for _ in range(2)]
        assert {frames[0] for frames in replies} == {b"search", b"watch"}
        assert all(response_adapter.validate_json(frames[-1]).kind != "error" for frames in replies)
    completed = [entry for entry in log_output.entries if entry["event"] == "request_complete"]
    assert len({entry["request_id"] for entry in completed}) == len(completed)
    assert all(entry["repo"] == fake_repo for entry in completed)
    assert completed[0]["kind"] == read_request.kind


@pytest.mark.parametrize("limit", [1, 2])
def test_read_capacity_queues_without_rejecting(
    limit: int,
    runtime_dir: Path,
    seeded_index: Index,
    stub_embedding_model: None,
    fake_repo: str,
    rpc_send: Callable[[Request], Response],
    rpc_peer: zmq.Socket[bytes],
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    monkeypatch.setattr(config, "max_concurrent_reads", limit)
    original = seeded_index.disk_size_bytes
    started = threading.Barrier(limit + 1)
    released = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0

    def disk_size_bytes() -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if not released.is_set():
                started.wait(timeout=10)
                assert released.wait(10)
            return original()
        finally:
            with lock:
                active -= 1

    mocker.patch.object(seeded_index, "disk_size_bytes", side_effect=disk_size_bytes)
    with (
        serving(Daemon(runtime_dir, seeded_index, idle_poll_interval=60)),
        ThreadPoolExecutor(max_workers=limit) as clients,
    ):
        reads = [clients.submit(rpc_send, StatusRequest(repo_path=fake_repo)) for _ in range(limit)]
        try:
            started.wait(timeout=10)
            rpc_peer.send_multipart(
                [b"read", b"", StatusRequest(repo_path=fake_repo).model_dump_json().encode()]
            )
            # This independent write is an intake fence: the same peer's
            # earlier read has been received before this reply can arrive.
            rpc_peer.send_multipart(
                [
                    b"fence",
                    b"",
                    WatchRequest(repo_path=fake_repo, refs=["HEAD"]).model_dump_json().encode(),
                ]
            )
            frames = rpc_peer.recv_multipart()
            assert frames[:2] == [b"fence", b""]
            assert response_adapter.validate_json(frames[-1]).kind == "watch_set"
            assert peak == limit
        finally:
            released.set()
        assert all(read.result(timeout=5).kind == "status" for read in reads)
        frames = rpc_peer.recv_multipart()
        assert frames[:2] == [b"read", b""]
        assert response_adapter.validate_json(frames[-1]).kind == "status"
        assert peak == limit


def test_shutdown_errors_resource_waiters_before_running_work_finishes(
    held_query: None,
    held_watch: None,
    running_daemon: Daemon,
    seeded_index: Index,
    fake_repo: str,
    daemon_commit: str,
    query_started: threading.Event,
    query_released: threading.Event,
    write_started: threading.Event,
    write_released: threading.Event,
    rpc_send: Callable[[Request], Response],
    rpc_peer: zmq.Socket[bytes],
) -> None:
    with ThreadPoolExecutor(max_workers=3) as clients:
        active = clients.submit(rpc_send, SearchRequest(repo_path=fake_repo, query="load_config"))
        writer = clients.submit(rpc_send, WatchRequest(repo_path=fake_repo, refs=["main"]))
        try:
            assert query_started.wait(5)
            assert write_started.wait(5)
            requests: list[tuple[bytes, Request]] = [
                (b"search", SearchRequest(repo_path=fake_repo, query="Application")),
                (b"watch", WatchRequest(repo_path=fake_repo, refs=[daemon_commit])),
                (b"fence", DaemonConfigRequest()),
            ]
            for tag, request in requests:
                rpc_peer.send_multipart([tag, b"", request.model_dump_json().encode()])
            frames = rpc_peer.recv_multipart()
            assert frames[:2] == [b"fence", b""]
            assert response_adapter.validate_json(frames[-1]).kind == "daemon_config"
            stop = clients.submit(rpc_send, ShutdownRequest())
            replies = [rpc_peer.recv_multipart() for _ in range(2)]
            assert {frames[0] for frames in replies} == {b"search", b"watch"}
            for frames in replies:
                response = response_adapter.validate_json(frames[-1])
                assert isinstance(response, ErrorResponse)
                assert response.code is ErrorCode.SHUTTING_DOWN
            assert not active.done()
            assert not writer.done()
            assert not stop.done()
        finally:
            query_released.set()
            write_released.set()
        assert active.result(timeout=5).kind == "search"
        assert writer.result(timeout=5).kind == "watch_set"
        assert stop.result(timeout=5).kind == "ok"
    assert daemon_commit not in seeded_index.watched_refs(seeded_index.repo(fake_repo))


def test_shutdown_removes_work_waiting_in_the_executor(
    runtime_dir: Path,
    seeded_index: Index,
    fake_repo: str,
    daemon_commit: str,
    rpc_send: Callable[[Request], Response],
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    monkeypatch.setattr(config, "embed", False)
    monkeypatch.setattr(config, "reranker_model", "")
    monkeypatch.setattr(config, "warmup", False)
    released = threading.Event()
    discovery = mocker.spy(daemon_rpc, "normalise_repo_path")

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        occupied = asyncio.Event()
        server = Daemon(runtime_dir, seeded_index, idle_poll_interval=60)
        with (
            ThreadPoolExecutor(max_workers=1) as executor,
            ThreadPoolExecutor(max_workers=2) as clients,
        ):
            loop.set_default_executor(executor)
            serving_task = asyncio.create_task(server.serve())
            try:
                assert await asyncio.to_thread(server.wait_ready)

                def block() -> None:
                    loop.call_soon_threadsafe(occupied.set)
                    assert released.wait(20)

                blocker = loop.run_in_executor(None, block)
                await asyncio.wait_for(occupied.wait(), timeout=5)
                submitted = asyncio.Event()
                original_submit = executor.submit

                def submit[**P, T](
                    fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
                ) -> Future[T]:
                    future = original_submit(fn, *args, **kwargs)
                    submitted.set()
                    return future

                mocker.patch.object(executor, "submit", side_effect=submit)
                write = loop.run_in_executor(
                    clients, rpc_send, WatchRequest(repo_path=fake_repo, refs=[daemon_commit])
                )
                await asyncio.wait_for(submitted.wait(), timeout=5)
                stop = loop.run_in_executor(clients, rpc_send, ShutdownRequest())
                response = await asyncio.wait_for(asyncio.shield(write), timeout=5)
                assert isinstance(response, ErrorResponse)
                assert response.code is ErrorCode.SHUTTING_DOWN
                assert not stop.done()
                released.set()
                await blocker
                assert (await stop).kind == "ok"
            finally:
                released.set()
                server.request_shutdown()
                await serving_task

    asyncio.run(scenario())
    discovery.assert_not_called()
    assert daemon_commit not in seeded_index.watched_refs(seeded_index.repo(fake_repo))


@parametrize_with_cases("mutation", cases=".cases_rpc_mutation")
def test_read_overlapping_registry_changes_returns_a_result_or_domain_error(
    mutation: ForgetRequest | GcRequest,
    held_query: None,
    running_daemon: Daemon,
    seeded_index: Index,
    fake_repo: str,
    query_started: threading.Event,
    query_released: threading.Event,
    rpc_send: Callable[[Request], Response],
) -> None:
    with ThreadPoolExecutor(max_workers=2) as clients:
        read = clients.submit(rpc_send, SearchRequest(repo_path=fake_repo, query="load_config"))
        try:
            assert query_started.wait(5)
            response = clients.submit(rpc_send, mutation).result(timeout=10)
            assert not isinstance(response, ErrorResponse)
            assert not read.done()
        finally:
            query_released.set()
        response = read.result(timeout=5)
        if isinstance(response, ErrorResponse):
            assert response.code in {
                ErrorCode.REPO_NOT_FOUND,
                ErrorCode.INDEX_NOT_BUILT,
                ErrorCode.INDEX_IN_PROGRESS,
            }
        else:
            assert isinstance(response, SearchResponse)


def test_disconnected_search_finishes_without_blocking_other_clients(
    held_query: None,
    running_daemon: Daemon,
    fake_repo: str,
    query_started: threading.Event,
    query_released: threading.Event,
    rpc_send: Callable[[Request], Response],
    log_output: structlog.testing.LogCapture,
) -> None:
    with zmq.Context() as context:
        client = context.socket(zmq.REQ)
        client.setsockopt(zmq.LINGER, 0)
        client.connect(running_daemon.rpc_addr)
        try:
            client.send(
                SearchRequest(repo_path=fake_repo, query="load_config").model_dump_json().encode()
            )
            assert query_started.wait(5)
            client.close()
            assert rpc_send(StatusRequest(repo_path=fake_repo)).kind == "status"
        finally:
            client.close()
            query_released.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if any(
                entry["event"] == "request_complete" and entry["kind"] == "search"
                for entry in log_output.entries
            ):
                break
            time.sleep(0.01)
        else:
            pytest.fail("disconnected search did not finish")
        assert rpc_send(StatusRequest(repo_path=fake_repo)).kind == "status"
