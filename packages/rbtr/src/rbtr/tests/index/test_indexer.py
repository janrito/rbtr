"""Standalone indexing and borrowed data lifetime."""

from __future__ import annotations

import asyncio
import selectors
import threading
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.git import normalise_repo_path
from rbtr.index.index import Index
from rbtr.index.indexer import Indexer


@pytest.fixture
def index_for_indexer(index: Index, git_repo: pygit2.Repository) -> Index:
    index.register(normalise_repo_path(git_repo.workdir))
    return index


@pytest.fixture
def wake_released() -> Generator[threading.Event]:
    released = threading.Event()
    yield released
    released.set()


@pytest.fixture
def pending_wake(
    indexer: Indexer, wake_released: threading.Event, mocker: MockerFixture
) -> Generator[Future[None]]:
    ready = threading.Event()
    set_event = asyncio.Event.set

    def held_set(event: asyncio.Event) -> None:
        if threading.current_thread().name.startswith("indexer-wake"):
            ready.set()
            assert wake_released.wait(20), "foreign event update was not released"
        set_event(event)

    def wake() -> None:
        try:
            indexer.wake()
        finally:
            ready.set()

    mocker.patch.object(asyncio.Event, "set", held_set)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer-wake") as caller:
        future = caller.submit(wake)
        try:
            assert ready.wait(10), "wake neither reached an event update nor returned"
            yield future
        finally:
            wake_released.set()
            future.result(timeout=10)


@pytest.fixture
def indexing_finished() -> threading.Event:
    return threading.Event()


@pytest.fixture
def running_indexer(
    indexer: Indexer,
    pending_wake: Future[None],
    wake_released: threading.Event,
    indexing_finished: threading.Event,
    mocker: MockerFixture,
) -> Generator[Future[None]]:
    observe_sleep = threading.Event()
    sleeping = threading.Event()
    selector = selectors.DefaultSelector()
    select = selector.select

    def observed_select(timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        if timeout is None and observe_sleep.is_set():
            sleeping.set()
        return select(timeout)

    mocker.patch.object(selector, "select", observed_select)
    loop = asyncio.SelectorEventLoop(selector)
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    future = asyncio.run_coroutine_threadsafe(indexer.run(), loop)
    future.add_done_callback(lambda _future: indexing_finished.set())
    loop.call_soon_threadsafe(observe_sleep.set)
    try:
        assert sleeping.wait(10), "indexer loop did not reach its selector wait"
        yield future
    finally:
        wake_released.set()
        loop.call_soon_threadsafe(lambda: None)
        indexer.request_shutdown()
        try:
            future.result(timeout=20)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=10)
            assert not thread.is_alive(), "indexer loop did not stop"
            loop.close()


@pytest.mark.parametrize("stopped", [False, True])
def test_initial_work_keeps_its_borrowed_index_open(
    indexer: Indexer,
    index_for_indexer: Index,
    git_repo: pygit2.Repository,
    stopped: bool,
) -> None:
    if stopped:
        indexer.request_shutdown()

    async def run() -> None:
        async with asyncio.timeout(20):
            await indexer.run()

    asyncio.run(run())
    at = index_for_indexer.repo(normalise_repo_path(git_repo.workdir)).at(str(git_repo.head.target))
    assert index_for_indexer.has_indexed(at=at) is not stopped
    if not stopped:
        counts = index_for_indexer.chunk_counts_for_snapshot(at=at)
        assert counts.total > 0
        assert counts.is_fully_embedded
    assert indexer.snapshot() is None


def test_a_wake_racing_startup_runs_stored_work(
    running_indexer: Future[None],
    pending_wake: Future[None],
    wake_released: threading.Event,
    indexing_finished: threading.Event,
    index_for_indexer: Index,
    git_repo: pygit2.Repository,
) -> None:
    wake_released.set()
    pending_wake.result(timeout=10)
    assert indexing_finished.wait(10), "startup wake left actual indexing asleep"
    running_indexer.result(timeout=10)
    at = index_for_indexer.repo(normalise_repo_path(git_repo.workdir)).at(str(git_repo.head.target))
    assert index_for_indexer.has_indexed(at=at)
    assert index_for_indexer.chunk_counts_for_snapshot(at=at).is_fully_embedded
