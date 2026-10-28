"""Status observes native progress independently of notification delivery."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
import zmq
import zmq.asyncio
from pytest_mock import MockerFixture

from rbtr.daemon.messages import (
    GcAllRequest,
    ProgressNotification,
    Request,
    Response,
    StatusRequest,
    StatusResponse,
    WatchRequest,
    notification_adapter,
)
from rbtr.daemon.server import Daemon
from rbtr.domain.models import GcMode
from rbtr.index.index import Index

from .conftest import serving


@pytest.fixture
def relay_held() -> threading.Event:
    return threading.Event()


@pytest.fixture
def relay_released() -> Generator[threading.Event]:
    released = threading.Event()
    yield released
    released.set()


@pytest.fixture
def held_progress(
    relay_held: threading.Event,
    relay_released: threading.Event,
    mocker: MockerFixture,
) -> None:
    receive = zmq.asyncio.Socket.recv

    async def recv(socket: zmq.asyncio.Socket) -> bytes:
        raw: bytes = await receive(socket)
        if socket.type == zmq.PULL and not relay_held.is_set():
            notification = notification_adapter.validate_json(raw)
            if isinstance(notification, ProgressNotification):
                relay_held.set()
                assert await asyncio.to_thread(relay_released.wait, 20), "relay was not released"
        return raw

    mocker.patch.object(zmq.asyncio.Socket, "recv", recv)


@pytest.fixture
def late_progress(
    fake_repo: str,
    second_repo: str,
    relay_held: threading.Event,
    relay_released: threading.Event,
    mocker: MockerFixture,
) -> None:
    receive = zmq.asyncio.Socket.recv
    delayed: bytes | None = None
    current_seen = False

    async def recv(socket: zmq.asyncio.Socket) -> bytes:
        nonlocal delayed, current_seen
        if delayed is not None and current_seen:
            assert await asyncio.to_thread(relay_released.wait, 20), "relay was not released"
            raw: bytes = delayed
            delayed = None
            return raw
        raw = await receive(socket)
        if socket.type == zmq.PULL:
            notification = notification_adapter.validate_json(raw)
            if isinstance(notification, ProgressNotification):
                if notification.repo_path == fake_repo and notification.phase == "finalising":
                    delayed = raw
                    relay_held.set()
                    return await recv(socket)
                if notification.repo_path == second_repo and notification.phase == "edges":
                    current_seen = True
        return raw

    mocker.patch.object(zmq.asyncio.Socket, "recv", recv)


@pytest.fixture
def running_daemon(
    runtime_dir: Path,
    empty_index: Index,
    stub_embedding_model: None,
    held_build: None,
    held_progress: None,
) -> Generator[Daemon]:
    with serving(
        Daemon(runtime_dir, empty_index, idle_poll_interval=60, busy_poll_interval=60)
    ) as daemon:
        yield daemon


@pytest.fixture
def late_progress_daemon(
    runtime_dir: Path,
    empty_index: Index,
    stub_embedding_model: None,
    held_build: None,
    late_progress: None,
) -> Generator[Daemon]:
    with serving(
        Daemon(runtime_dir, empty_index, idle_poll_interval=60, busy_poll_interval=60)
    ) as daemon:
        yield daemon


@pytest.fixture
def subscriber(
    late_progress_daemon: Daemon, rpc_send: Callable[[Request], Response]
) -> Generator[zmq.Socket[bytes]]:
    """Establish the subscription by receiving an actual collection notification."""
    with zmq.Context() as context, context.socket(zmq.SUB) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.subscribe(b"")
        socket.connect(late_progress_daemon.pub_addr)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            rpc_send(GcAllRequest(mode=GcMode.WATCHED))
            if socket.poll(100):
                socket.recv()
                break
        else:
            pytest.fail("subscription did not receive a notification")
        yield socket


def test_status_sees_progress_before_the_relay_receives_it(
    running_daemon: Daemon,
    held_progress: None,
    second_repo: str,
    rpc_send: Callable[[Request], Response],
    write_started: threading.Event,
    write_released: threading.Event,
    relay_held: threading.Event,
    relay_released: threading.Event,
) -> None:
    try:
        rpc_send(WatchRequest(repo_path=second_repo))
        assert write_started.wait(10), "native build did not report progress"
        assert relay_held.wait(10), "progress did not reach PULL"
        status = rpc_send(StatusRequest(repo_path=second_repo))
        assert isinstance(status, StatusResponse)
        assert status.active_build is not None
        assert status.active_build.phase == "edges"
        assert status.active_build.current == status.active_build.total == 0
        assert status.active_embed is None
    finally:
        relay_released.set()
        write_released.set()


def test_an_earlier_runs_delivery_keeps_the_current_runs_progress(
    late_progress_daemon: Daemon,
    late_progress: None,
    fake_repo: str,
    second_repo: str,
    rpc_send: Callable[[Request], Response],
    subscriber: zmq.Socket[bytes],
    write_started: threading.Event,
    write_released: threading.Event,
    relay_held: threading.Event,
    relay_released: threading.Event,
) -> None:
    try:
        rpc_send(WatchRequest(repo_path=fake_repo))
        assert relay_held.wait(10), "earlier run did not reach PULL"
        rpc_send(WatchRequest(repo_path=second_repo))
        assert write_started.wait(10), "later native build did not report progress"
        while True:
            notification = notification_adapter.validate_json(subscriber.recv())
            if (
                isinstance(notification, ProgressNotification)
                and notification.repo_path == second_repo
                and notification.phase == "edges"
            ):
                break
        relay_released.set()
        while True:
            notification = notification_adapter.validate_json(subscriber.recv())
            if (
                isinstance(notification, ProgressNotification)
                and notification.repo_path == fake_repo
                and notification.phase == "finalising"
            ):
                break
        status = rpc_send(StatusRequest(repo_path=second_repo))
        assert isinstance(status, StatusResponse)
        assert status.active_build is not None
        assert status.active_build.repo_path == second_repo
        assert status.active_build.phase == "edges"
        assert status.active_build.current == status.active_build.total == 0
        assert status.active_embed is None
    finally:
        relay_released.set()
        write_released.set()
