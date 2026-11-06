"""Tests for the ZMQ daemon server and sync client.

Uses shared fixtures from conftest.py. Tests exercise the full
send → dispatch → respond path through the actual ZMQ socket.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Generator
from pathlib import Path

import duckdb
import pytest
import structlog
import zmq
from pytest_cases import parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import (
    CollectWork,
    DaemonConfigRequest,
    EmbedWork,
    ErrorCode,
    ErrorResponse,
    GcAllRequest,
    GcEndedNotification,
    IndexState,
    MigrateWork,
    Notification,
    OkResponse,
    ProgressNotification,
    ReadSymbolRequest,
    ShutdownRequest,
    StatusAllRequest,
    StatusRequest,
    StatusResponse,
    notification_adapter,
    response_adapter,
)
from rbtr.daemon.server import Daemon
from rbtr.domain.models import GcMode
from rbtr.errors import DaemonBusyError, RbtrError
from rbtr.index.constants import SCHEMA_VERSION
from rbtr.index.embeddings import Embedder, EmbedResult
from rbtr.index.index import Index
from rbtr.index.migrations import Migrate
from rbtr.index.progress import ProgressCallback, RepoProgressCallback
from rbtr.tests.daemon.cases_server import MalformedMessage, StopScenario

from .conftest import serving, subscribed


@parametrize_with_cases("scenario", cases=".cases_server")
def test_how_a_daemon_stops(
    scenario: StopScenario, log_output: structlog.testing.LogCapture
) -> None:
    """What stops a serving daemon, what `serve()` raises, and what it logs."""
    raised: list[BaseException] = []

    def serve() -> None:
        try:
            asyncio.run(scenario.server.serve())
        except BaseException as exc:  # noqa: BLE001 - CancelledError is a BaseException
            raised.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        assert scenario.server.wait_ready(), "daemon did not start within timeout"
        scenario.stop()
        thread.join(timeout=10)
        assert not thread.is_alive(), "the daemon did not stop"
    finally:
        if thread.is_alive():
            scenario.server.request_shutdown()
            thread.join(timeout=30)
            assert not thread.is_alive(), "the daemon did not stop"

    assert [str(exc) for exc in raised] == scenario.raised
    assert [e["event"] for e in log_output.entries if e["log_level"] == "error"] == scenario.logged


def test_multiple_requests(running_daemon: Daemon, fake_repo: str) -> None:
    with DaemonClient(running_daemon.runtime_dir) as client:
        r1 = client.send(StatusRequest(repo_path=fake_repo))
        r2 = client.send(StatusRequest(repo_path=fake_repo))
    assert r1.kind == "status"
    assert r2.kind == "status"


def test_gc_publishes_its_start_progress_and_end(running_daemon: Daemon, fake_repo: str) -> None:
    """A gc over every repo publishes its start and one progress per repo as progress, then its end."""
    with subscribed(running_daemon, fake_repo) as sub:
        with DaemonClient(running_daemon.runtime_dir) as client:
            client.send(GcAllRequest(mode=GcMode.WATCHED))
        published: list[Notification] = []
        while sub.poll(2000):
            # The gc's notifications; others, such as an embed's progress, are skipped.
            match notification_adapter.validate_json(sub.recv()):
                case GcEndedNotification() | ProgressNotification(work=CollectWork()) as gc:
                    published.append(gc)

    assert published == [
        ProgressNotification(work=CollectWork(), phase="collecting", current=0, total=0),
        ProgressNotification(work=CollectWork(), phase="collecting", current=1, total=1),
        GcEndedNotification(),
    ]


@pytest.mark.usefixtures("held_gc")
def test_status_lists_a_running_gc_for_every_repository(
    running_daemon: Daemon,
    fake_repo: str,
    second_repo: str,
    gc_started: threading.Event,
    gc_released: threading.Event,
) -> None:
    """A gc concerns every repository, so a status for any of them lists it while it runs."""
    gc = threading.Thread(
        target=lambda: DaemonClient(running_daemon.runtime_dir).send(
            GcAllRequest(mode=GcMode.WATCHED)
        )
    )
    gc.start()
    try:
        assert gc_started.wait(10), "gc did not start"
        with DaemonClient(running_daemon.runtime_dir) as client:
            statuses = [
                client.send(StatusRequest(repo_path=fake_repo)),
                client.send(StatusRequest(repo_path=second_repo)),
                client.send(StatusAllRequest()),
            ]
    finally:
        gc_released.set()
        gc.join(10)
    for status in statuses:
        assert isinstance(status, StatusResponse)
        [job] = [job for job in status.active if isinstance(job.work, CollectWork)]
        assert (job.phase, job.current, job.total) == ("collecting", 1, 2)


def test_status_lists_the_gc_maintenance_runs(
    runtime_dir: Path,
    seeded_index: Index,
    fake_repo: str,
    gc_started: threading.Event,
    gc_released: threading.Event,
    stub_embedding_model: None,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Maintenance's gc is tracked as a request's is, so status lists it while it runs."""

    def maintain(*_args: object, on_progress: RepoProgressCallback, **_kwargs: object) -> None:
        on_progress(fake_repo, 1, 2)
        gc_started.set()
        assert gc_released.wait(20), "maintenance was not released"

    mocker.patch("rbtr.index.indexer.run_maintenance", maintain)
    monkeypatch.setattr(config, "maintenance_idle_s", 0.0)
    server = Daemon(runtime_dir, index=seeded_index, idle_poll_interval=0.2)
    with serving(server), DaemonClient(server.runtime_dir) as client:
        assert gc_started.wait(30), "maintenance did not start"
        status = client.send(StatusRequest(repo_path=fake_repo))
        gc_released.set()
    assert isinstance(status, StatusResponse)
    [job] = [job for job in status.active if isinstance(job.work, CollectWork)]
    assert (job.phase, job.current, job.total) == ("collecting", 1, 2)


@pytest.mark.usefixtures("held_gc")
def test_status_lists_an_embed_and_a_gc_running_together(
    runtime_dir: Path,
    seeded_index: Index,
    fake_repo: str,
    gc_started: threading.Event,
    gc_released: threading.Event,
    stub_embedding_model: None,
    mocker: MockerFixture,
) -> None:
    """A gc runs while an embed waits on its model, so status lists both."""
    embedding = threading.Event()
    embed = Embedder.embed

    def held_embed(embedder: Embedder, texts: list[str]) -> list[EmbedResult]:
        embedding.set()
        assert gc_released.wait(20), "the embed was not released"
        return embed(embedder, texts)

    mocker.patch.object(Embedder, "embed", held_embed)
    server = Daemon(runtime_dir, index=seeded_index)
    with serving(server):
        assert embedding.wait(20), "the embed did not start"
        gc = threading.Thread(
            target=lambda: DaemonClient(server.runtime_dir).send(GcAllRequest(mode=GcMode.WATCHED))
        )
        gc.start()
        try:
            assert gc_started.wait(10), "gc did not start"
            with DaemonClient(server.runtime_dir) as client:
                status = client.send(StatusRequest(repo_path=fake_repo))
        finally:
            gc_released.set()
            gc.join(10)
    assert isinstance(status, StatusResponse)
    assert {type(job.work) for job in status.active} == {EmbedWork, CollectWork}


def test_a_gc_that_fails_still_ends(
    running_daemon: Daemon, fake_repo: str, mocker: MockerFixture
) -> None:
    """A failing gc publishes its end and leaves status, as one that succeeds does."""
    mocker.patch("rbtr.daemon.server.handle_gc_all", side_effect=RbtrError("the gc failed"))
    with subscribed(running_daemon, fake_repo) as sub:
        with DaemonClient(running_daemon.runtime_dir) as client:
            reply = client.send(GcAllRequest(mode=GcMode.WATCHED))
            status = client.send(StatusAllRequest())
        published: list[Notification] = []
        while sub.poll(2000):
            # The gc's notifications; others, such as an embed's progress, are skipped.
            match notification_adapter.validate_json(sub.recv()):
                case GcEndedNotification() | ProgressNotification(work=CollectWork()) as gc:
                    published.append(gc)

    assert isinstance(reply, ErrorResponse)
    assert published == [
        ProgressNotification(work=CollectWork(), phase="collecting", current=0, total=0),
        GcEndedNotification(),
    ]
    assert isinstance(status, StatusResponse)
    assert not [job for job in status.active if isinstance(job.work, CollectWork)]


@pytest.fixture
def migration_sql() -> str:
    """The project migration `migrating_index` waits for; a test parametrises it to override."""
    return "ALTER TABLE chunks ADD COLUMN note TEXT;"


@pytest.fixture
def migrating_index(
    tmp_path: Path, fake_repo: str, chains: Path, migration_sql: str, mocker: MockerFixture
) -> Generator[Index]:
    """An on-disk index opened with `Migrate.LATER`, its one project file waiting for 2099.1.0."""
    path = tmp_path / "data" / "index.duckdb"
    seed = Index(path, writable=True)
    seed.register(fake_repo)
    seed.close()
    (chains / "project").mkdir()
    (chains / "project" / f"{SCHEMA_VERSION}__2099.1.0.sql").write_text(migration_sql)
    mocker.patch("rbtr.index.store.SCHEMA_VERSION", "2099.1.0")
    index = Index(path, writable=True, migrate=Migrate.LATER)
    yield index
    index.close()


def test_a_daemon_answers_while_it_migrates_and_indexes_after(
    runtime_dir: Path,
    migrating_index: Index,
    fake_repo: str,
    stub_embedding_model: None,
    mocker: MockerFixture,
) -> None:
    """The daemon is ready before it migrates; meanwhile status says so and reads reply `index_in_progress`."""
    held, released = threading.Event(), threading.Event()
    migrate = Index.migrate

    def held_migrate(index: Index, on_progress: ProgressCallback) -> None:
        held.set()
        assert released.wait(20), "the migration was not released"
        migrate(index, on_progress)

    mocker.patch.object(Index, "migrate", held_migrate)
    server = Daemon(runtime_dir, index=migrating_index, idle_poll_interval=0.2)
    with serving(server), DaemonClient(server.runtime_dir) as client:
        assert held.wait(10), "the migration did not start"
        read = client.send(ReadSymbolRequest(repo_path=fake_repo, symbol="load_config"))
        status = client.send(StatusAllRequest())
        released.set()
        assert isinstance(read, ErrorResponse)
        assert read.code is ErrorCode.INDEX_IN_PROGRESS
        assert isinstance(status, StatusResponse)
        assert status.state is IndexState.MIGRATING
        assert [type(job.work) for job in status.active] == [MigrateWork]
        deadline = time.monotonic() + 30
        while True:
            after = client.send(StatusRequest(repo_path=fake_repo))
            if isinstance(after, StatusResponse) and after.state is IndexState.INDEXED:
                break
            assert time.monotonic() < deadline, f"the daemon did not index after migrating: {after}"
            time.sleep(0.1)


@pytest.mark.parametrize("migration_sql", ["SELECT * FROM no_such_table;"])
def test_a_failing_migration_stops_the_daemon_and_keeps_the_old_file(
    runtime_dir: Path, migrating_index: Index, tmp_path: Path
) -> None:
    """The daemon stops with the migration's error; the registry still names the file it had."""
    (old_file,) = (tmp_path / "data" / "index").glob("*.duckdb")
    server = Daemon(runtime_dir, index=migrating_index)
    failure: list[BaseException] = []

    def serve() -> None:
        try:
            asyncio.run(server.serve())
        except duckdb.Error as exc:
            failure.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert server.wait_ready(), "daemon did not start"
    thread.join(30)
    assert not thread.is_alive(), "daemon did not stop"
    assert [type(exc) for exc in failure] == [duckdb.CatalogException]
    assert list((tmp_path / "data" / "index").glob("*.duckdb")) == [old_file]


def test_an_idle_daemon_runs_maintenance(
    runtime_dir: Path,
    seeded_index: Index,
    fake_repo: str,
    daemon_commit: str,
    stub_embedding_model: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon idle for `maintenance_idle_s` runs a pass, published as a gc.

    The pass records the registered checkout whose path is gone as missing,
    which starts its grace period, collects the live repo, and rebuilds the
    FTS index no write changed.
    """
    # Held off until the subscription is confirmed, so none of the pass is missed.
    monkeypatch.setattr(config, "maintenance_idle_s", float("inf"))
    gone = seeded_index.register(str(tmp_path / "gone"))
    live = seeded_index.repo(fake_repo)
    with seeded_index.reader(live) as cur:
        cur.execute("PRAGMA drop_fts_index('chunks')")
    server = Daemon(
        runtime_dir, index=seeded_index, idle_poll_interval=1.0, busy_poll_interval=60.0
    )
    with serving(server), subscribed(server, fake_repo) as sub:
        monkeypatch.setattr(config, "maintenance_idle_s", 0.0)
        published: list[Notification] = []
        while sub.poll(30_000):
            # The pass's notifications; others, such as an embed's progress, are skipped.
            match notification_adapter.validate_json(sub.recv()):
                case GcEndedNotification() as gc:
                    published.append(gc)
                    break
                case ProgressNotification(work=CollectWork()) as gc:
                    published.append(gc)

    assert published == [
        ProgressNotification(work=CollectWork(), phase="collecting", current=0, total=0),
        ProgressNotification(work=CollectWork(), phase="collecting", current=1, total=1),
        GcEndedNotification(),
    ]
    assert gone in seeded_index.missing_since()
    assert len(seeded_index.fulltext_matches("load_config", within=[live.at(daemon_commit)])) > 0


@parametrize_with_cases("message", cases=".cases_server", prefix="message_")
def test_garbage_returns_error(
    message: MalformedMessage, running_daemon: Daemon, fake_repo: str
) -> None:
    message.client.send_multipart(message.frames)
    frames = message.client.recv_multipart()
    assert frames[:-1] == message.reply_prefix
    response = response_adapter.validate_json(frames[-1])
    assert isinstance(response, ErrorResponse)
    assert response.code is ErrorCode.INVALID_REQUEST
    message.client.send_multipart(
        [*message.reply_prefix, StatusRequest(repo_path=fake_repo).model_dump_json().encode()]
    )
    frames = message.client.recv_multipart()
    assert frames[:-1] == message.reply_prefix
    assert response_adapter.validate_json(frames[-1]).kind == "status"


def test_message_without_return_envelope_does_not_stop_service(
    running_daemon: Daemon, fake_repo: str, rpc_peer: zmq.Socket[bytes]
) -> None:
    rpc_peer.send(b"no-delimiter")
    rpc_peer.send_multipart(
        [b"return-hop", b"", StatusRequest(repo_path=fake_repo).model_dump_json().encode()]
    )
    frames = rpc_peer.recv_multipart()
    assert frames[:-1] == [b"return-hop", b""]
    assert response_adapter.validate_json(frames[-1]).kind == "status"


def test_malformed_argument_returns_field_feedback(running_daemon: Daemon) -> None:
    """A structurally-invalid argument is rejected with per-field detail.

    The error names the offending field and echoes the received value,
    so the caller can see how the argument was mis-shaped — general
    feedback, not a hand-picked failure mode.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"ipc://{running_daemon.runtime_dir / 'daemon.rpc'}")
    # file_paths must be a list of strings; element 123 is not a string.
    sock.send(b'{"kind":"read_symbol","repo_path":"/r","symbol":"x","file_paths":[123]}')
    raw = sock.recv()
    resp = response_adapter.validate_json(raw)
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.INVALID_REQUEST
    assert "file_paths" in resp.message
    assert "123" in resp.message
    sock.close()
    ctx.term()


def test_json_encoded_list_is_decoded_then_validated(running_daemon: Daemon) -> None:
    """A JSON-encoded list arg is decoded, then validated by pydantic.

    The unwrap only decodes; it does not type-check. A decoded list of
    the wrong element type is therefore rejected by normal validation
    with field-level feedback — no duplicated checking.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"ipc://{running_daemon.runtime_dir / 'daemon.rpc'}")
    # file_paths delivered as a JSON string encoding a list of ints.
    sock.send(b'{"kind":"read_symbol","repo_path":"/r","symbol":"x","file_paths":["[1, 2]"]}')
    raw = sock.recv()
    resp = response_adapter.validate_json(raw)
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.INVALID_REQUEST
    assert "file_paths" in resp.message
    sock.close()
    ctx.term()


def test_non_git_repo_path_returns_error_and_daemon_survives(
    running_daemon: Daemon, fake_repo: str, tmp_path: Path
) -> None:
    """A request with a non-git repo_path must not crash the daemon.

    Regression: `normalise_repo_path` raised `RbtrError` outside the
    dispatch error handler, so a client whose cwd was not a git repo
    took the whole (shared) daemon down mid-job.
    """
    not_git = tmp_path / "not_a_repo"
    not_git.mkdir()

    with DaemonClient(running_daemon.runtime_dir) as client:
        bad = client.send(StatusRequest(repo_path=str(not_git)))
        assert isinstance(bad, ErrorResponse)
        assert bad.code == ErrorCode.REPO_NOT_FOUND

        # Daemon is still alive and serving other repos.
        good = client.send(StatusRequest(repo_path=fake_repo))
    assert good.kind == "status"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ValueError, ErrorCode.INTERNAL),
        (duckdb.OutOfMemoryException, ErrorCode.OUT_OF_MEMORY),
    ],
)
def test_handler_exception_returns_error(
    running_daemon: Daemon, error: type[Exception], code: ErrorCode, mocker: MockerFixture
) -> None:
    mocker.patch("rbtr.daemon.handlers.get_version", side_effect=error("handler broke"))

    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send(DaemonConfigRequest())
    assert isinstance(resp, ErrorResponse)
    assert resp.code == code
    assert "handler broke" in resp.message


def test_connection_refused(runtime_dir: Path) -> None:
    with DaemonClient(runtime_dir) as client, pytest.raises(DaemonBusyError):
        client.send(ShutdownRequest())


def test_send_or_raise_on_success(running_daemon: Daemon) -> None:
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send_or_raise(ShutdownRequest())
    assert isinstance(resp, OkResponse)


def test_send_or_raise_on_error(running_daemon: Daemon, mocker: MockerFixture) -> None:
    mocker.patch("rbtr.daemon.handlers.get_version", side_effect=ValueError("boom"))

    with (
        DaemonClient(running_daemon.runtime_dir) as client,
        pytest.raises(RbtrError, match="boom"),
    ):
        client.send_or_raise(DaemonConfigRequest())


def test_send_or_raise_as_narrows_response(running_daemon: Daemon) -> None:
    with DaemonClient(running_daemon.runtime_dir) as client:
        resp = client.send_or_raise_as(OkResponse, ShutdownRequest())
    # mypy would reject a .foo on resp if narrowing didn't work.
    assert resp.kind == "ok"


def test_send_or_raise_as_rejects_mismatched_response(
    running_daemon: Daemon,
) -> None:
    with (
        DaemonClient(running_daemon.runtime_dir) as client,
        pytest.raises(RbtrError, match=r"expected StatusResponse.*got OkResponse"),
    ):
        client.send_or_raise_as(StatusResponse, ShutdownRequest())
