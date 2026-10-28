"""Watcher polling uses configured idle and active-work intervals."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from queue import Queue

import pytest
from pytest_cases import parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import WatchRequest
from rbtr.daemon.server import Daemon
from rbtr.index.index import Index
from rbtr.index.indexer import Indexer, IndexerOutput

from .conftest import serving


@pytest.fixture
def service(empty_index: Index) -> Indexer:
    async def output(_event: IndexerOutput) -> None:
        pass

    return Indexer(
        empty_index,
        embedder=None,
        gpu_lock=asyncio.Lock(),
        write_sem=asyncio.Semaphore(1),
        stop=threading.Event(),
        on_output=output,
    )


def test_idle_interval_when_no_job_is_active(service: Indexer) -> None:
    assert service.poll_interval() == config.idle_poll_interval


@parametrize_with_cases("idle,busy", cases=".cases_watcher_backoff")
def test_busy_interval_while_a_job_runs(
    idle: float | None,
    busy: float | None,
    runtime_dir: Path,
    empty_index: Index,
    second_repo: str,
    held_build: None,
    write_started: threading.Event,
    write_released: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    monkeypatch.setattr(config, "embed", False)
    monkeypatch.setattr(config, "reranker_model", None)
    requested: Queue[float] = Queue()
    pause = Indexer._pause

    async def record(service: Indexer, delay: float) -> None:
        requested.put(delay)
        await pause(service, 0.01)

    mocker.patch.object(Indexer, "_pause", record)
    with serving(
        Daemon(runtime_dir, empty_index, idle_poll_interval=idle, busy_poll_interval=busy)
    ):
        try:
            assert requested.get(timeout=10) == (
                config.idle_poll_interval if idle is None else idle
            )
            with DaemonClient(runtime_dir) as client:
                client.send(WatchRequest(repo_path=second_repo))
            assert write_started.wait(10), "native build did not start"
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if requested.get(timeout=10) == (
                    config.busy_poll_interval if busy is None else busy
                ):
                    break
            else:
                pytest.fail("watcher did not request the busy interval")
        finally:
            write_released.set()
