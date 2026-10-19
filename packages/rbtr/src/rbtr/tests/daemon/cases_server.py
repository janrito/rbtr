"""Cases for how a serving daemon stops.

Each case builds a daemon, says what makes it stop, and says what
stopping shows: the errors `serve()` raises and the errors it logs.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pygit2
from pytest_mock import MockerFixture

from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import OkResponse, ShutdownRequest, WatchRequest
from rbtr.daemon.server import DaemonServer
from rbtr.domain.models import Chunk
from rbtr.git import FileEntry
from rbtr.index import build
from rbtr.index.index import Index


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
    server = DaemonServer(runtime_dir, index=None, idle_poll_interval=60.0, busy_poll_interval=60.0)

    def stop() -> None:
        with DaemonClient(runtime_dir) as client:
            assert isinstance(client.send(ShutdownRequest()), OkResponse)

    return StopScenario(server=server, stop=stop, raised=[], logged=[])


def case_shutdown_during_warmup(runtime_dir: Path, registered_index: Index) -> StopScenario:
    """A shutdown while the embedder warms up cancels the warmup; the
    cancellation does not escape `serve()`."""
    server = DaemonServer(
        runtime_dir, index=registered_index, idle_poll_interval=60.0, busy_poll_interval=60.0
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


def case_shutdown_during_build(
    runtime_dir: Path, git_repo: pygit2.Repository, index: Index, mocker: MockerFixture
) -> StopScenario:
    """A shutdown during a build ends it after the file in hand, as a stop
    rather than a failure: no error is published or logged."""
    with index.session() as ws:
        ws.register_repo(git_repo.workdir)
    server = DaemonServer(
        runtime_dir, index=index, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    extract = build.extract_file

    def slow_extract(entry: FileEntry, language: str) -> list[Chunk]:
        time.sleep(3)  # five files: a build longer than the test waits
        return extract(entry, language)

    mocker.patch.object(build, "extract_file", side_effect=slow_extract)

    def stop() -> None:
        with DaemonClient(runtime_dir) as client:
            client.send(WatchRequest(repo_path=git_repo.workdir, refs=["HEAD"]))
            while server._active_build is None:
                time.sleep(0.05)
            client.send(ShutdownRequest())

    return StopScenario(server=server, stop=stop, raised=[], logged=[])


def case_job_worker_fails(
    runtime_dir: Path, seeded_index: Index, fake_repo: str, mocker: MockerFixture
) -> StopScenario:
    """An error the job worker does not expect stops the daemon, logged,
    rather than leaving it answering reads with nothing being built."""
    server = DaemonServer(
        runtime_dir, index=seeded_index, idle_poll_interval=60.0, busy_poll_interval=60.0
    )
    mocker.patch.object(server, "_find_next_job", side_effect=RuntimeError("unexpected"))

    def stop() -> None:
        # A watch wakes the worker, which looks for its next job.
        with DaemonClient(runtime_dir) as client:
            client.send(WatchRequest(repo_path=fake_repo, refs=["HEAD"]))

    return StopScenario(server=server, stop=stop, raised=["unexpected"], logged=["task_failed"])


def case_watcher_fails(runtime_dir: Path, mocker: MockerFixture) -> StopScenario:
    """An error the watcher does not expect stops the daemon, logged,
    rather than leaving it answering reads with nothing being watched."""
    server = DaemonServer(runtime_dir, index=None, idle_poll_interval=60.0, busy_poll_interval=60.0)
    mocker.patch.object(server, "_next_poll_interval", side_effect=RuntimeError("unexpected"))
    return StopScenario(
        server=server, stop=lambda: None, raised=["unexpected"], logged=["task_failed"]
    )
