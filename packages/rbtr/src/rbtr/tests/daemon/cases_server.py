"""Cases for how a serving daemon stops.

Each case builds a daemon, says what makes it stop, and says what
stopping shows: the errors `serve()` raises and the errors it logs.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pytest_mock import MockerFixture

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import OkResponse, ShutdownRequest, WatchRequest
from rbtr.daemon.server import DaemonServer
from rbtr.index.store import IndexStore


@dataclass(frozen=True)
class StopScenario:
    """A daemon, what makes it stop once it serves, and what stopping shows.

    `raised` holds the messages of the errors `serve()` raises, and
    `logged` the events it logs at error level.
    """

    server: DaemonServer
    stop: Callable[[], None]
    raised: list[str]
    logged: list[str]


def case_shutdown_request(runtime_dir: Path) -> StopScenario:
    """A shutdown request is answered, and the daemon stops cleanly."""
    server = DaemonServer(runtime_dir, store=None, idle_poll_interval=60.0, busy_poll_interval=60.0)

    def stop() -> None:
        with DaemonClient(runtime_dir) as client:
            assert isinstance(client.send(ShutdownRequest()), OkResponse)

    return StopScenario(server=server, stop=stop, raised=[], logged=[])


def case_shutdown_during_warmup(runtime_dir: Path, unindexed_store: IndexStore) -> StopScenario:
    """A shutdown while the embedder warms up cancels the warmup; the
    cancellation does not escape `serve()`."""
    server = DaemonServer(
        runtime_dir, store=unindexed_store, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    server._warmup = True
    server._reranker = None
    started = threading.Event()
    release = threading.Event()

    def blocking_warmup() -> None:
        started.set()
        release.wait(timeout=10)

    assert server._embedder is not None
    server._embedder.warmup = blocking_warmup  # type: ignore[method-assign]  # stub holds warmup in flight

    def stop() -> None:
        assert started.wait(timeout=2), "warmup task did not start"
        with DaemonClient(runtime_dir) as client:
            client.send(ShutdownRequest())
        release.set()

    return StopScenario(server=server, stop=stop, raised=[], logged=[])


def case_job_worker_fails(
    runtime_dir: Path, seeded_store: IndexStore, fake_repo: str, mocker: MockerFixture
) -> StopScenario:
    """An error the job worker does not expect stops the daemon, logged,
    rather than leaving it answering reads with nothing being built."""
    server = DaemonServer(
        runtime_dir, store=seeded_store, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    mocker.patch.object(server, "_find_next_job", side_effect=RuntimeError("unexpected"))

    def stop() -> None:
        # A watch wakes the worker, which looks for its next job.
        with DaemonClient(runtime_dir) as client:
            client.send(WatchRequest(repo_path=fake_repo, refs=["HEAD"]))

    return StopScenario(
        server=server, stop=stop, raised=["unexpected"], logged=["job_worker_failed"]
    )
