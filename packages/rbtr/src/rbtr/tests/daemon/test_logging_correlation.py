"""Request and job correlation: the daemon binds correlation keys into
the logging context so a request or background job can be followed
across modules.

Asserts on the autouse `log_output` (`LogCapture`), which includes
`merge_contextvars`, so contextvar binds surface on captured events.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
import structlog

from rbtr.daemon.messages import OkResponse, StatusRequest
from rbtr.daemon.server import DaemonServer
from rbtr.index.store import IndexStore

from .conftest import serving


@pytest.fixture
def server(runtime_dir: Path, seeded_store: IndexStore) -> DaemonServer:
    """A daemon server over the seeded store (no event loop running)."""
    return DaemonServer(runtime_dir, seeded_store)


def _dispatch_status(server: DaemonServer, repo_path: str) -> None:
    raw = StatusRequest(repo_path=repo_path).model_dump_json().encode()
    asyncio.run(server._dispatch(raw))


def test_dispatch_binds_a_unique_context_per_request(
    server: DaemonServer,
    fake_repo: str,
    log_output: structlog.testing.LogCapture,
) -> None:
    async def handler(_req: object) -> OkResponse:
        structlog.get_logger("t").info("handler_ran")
        return OkResponse()

    server._handlers["status"] = handler
    _dispatch_status(server, fake_repo)
    _dispatch_status(server, fake_repo)

    ran = [e for e in log_output.entries if e["event"] == "handler_ran"]
    assert len(ran) == 2
    assert all("request_id" in e for e in ran)
    assert all(e["kind"] == "status" for e in ran)
    assert all(e["repo"] for e in ran)
    assert ran[0]["request_id"] != ran[1]["request_id"]


def test_binding_survives_to_thread(
    server: DaemonServer,
    fake_repo: str,
    log_output: structlog.testing.LogCapture,
) -> None:
    def in_thread() -> OkResponse:
        structlog.get_logger("t").info("in_thread")
        return OkResponse()

    async def handler(_req: object) -> OkResponse:
        return await asyncio.to_thread(in_thread)

    server._handlers["status"] = handler
    _dispatch_status(server, fake_repo)

    threaded = [e for e in log_output.entries if e["event"] == "in_thread"]
    assert len(threaded) == 1
    assert "request_id" in threaded[0]


def test_binding_is_scoped_to_the_request(server: DaemonServer, fake_repo: str) -> None:
    # FileSnapshot the context *inside the same task* as the dispatch: a
    # scoped (`bound_contextvars`) binding is gone once `_dispatch`
    # returns; an unscoped `bind_contextvars` would leak it here.
    async def dispatch_then_snapshot() -> dict[str, object]:
        raw = StatusRequest(repo_path=fake_repo).model_dump_json().encode()
        await server._dispatch(raw)
        return dict(structlog.contextvars.get_contextvars())

    assert asyncio.run(dispatch_then_snapshot()) == {}


def test_request_complete_carries_elapsed_ms(
    server: DaemonServer,
    fake_repo: str,
    log_output: structlog.testing.LogCapture,
) -> None:
    _dispatch_status(server, fake_repo)

    done = [e for e in log_output.entries if e["event"] == "request_complete"]
    assert len(done) == 1
    assert isinstance(done[0]["elapsed_ms"], float)
    assert done[0]["elapsed_ms"] >= 0


def test_embed_job_logs_are_correlated(
    running_daemon: DaemonServer,
    log_output: structlog.testing.LogCapture,
) -> None:
    """End-to-end: a background embed job tags its logs with job context.

    No bespoke daemon needed: `seeded_store` holds indexed-but-unembedded
    chunks, so `_recover_pending_embeds` wakes the worker at
    construction, and the default daemon's embedder is the stub.  The
    server runs in a thread in this process, so the autouse
    `LogCapture` sees the worker's events.
    """
    deadline = time.monotonic() + 10.0
    embedded: list[structlog.typing.EventDict] = []
    while time.monotonic() < deadline:
        embedded = [e for e in log_output.entries if e["event"] == "embedded_chunks"]
        if embedded:
            break
        time.sleep(0.05)

    assert embedded, "no embedded_chunks log captured"
    entry = embedded[-1]
    assert "job_id" in entry
    assert entry["job_kind"] == "embed"
    assert entry["repo"]
    assert "ref" in entry
    assert isinstance(entry["elapsed_ms"], float)


def test_watched_ref_build_is_observable(
    runtime_dir: Path,
    unindexed_store: IndexStore,
    log_output: structlog.testing.LogCapture,
) -> None:
    """A watched ref flows from detection to indexed, fully correlated.

    The watcher loop logs `watched_ref_stale` (naming the ref/sha) and
    wakes the worker, whose build emits `index_complete` tagged with the
    build `job_id`/`job_kind`/`ref` — closing the gap where a build left
    no log line naming the SHA it indexed.
    """
    # Its own server, not `running_daemon`: this test needs a watcher
    # fast enough to notice the stale ref, and the module's
    # `running_daemon` is the embed-job one.
    server = DaemonServer(
        runtime_dir,
        store=unindexed_store,
        idle_poll_interval=0.05,
        busy_poll_interval=0.05,
    )
    with serving(server):
        deadline = time.monotonic() + 10.0
        built: list[structlog.typing.EventDict] = []
        while time.monotonic() < deadline:
            built = [e for e in log_output.entries if e["event"] == "index_complete"]
            if built:
                break
            time.sleep(0.05)

    stale = [e for e in log_output.entries if e["event"] == "watched_ref_stale"]
    assert stale, "no watched_ref_stale log captured"
    assert stale[0]["ref"] == "HEAD"
    assert stale[0]["repo"]
    assert stale[0]["sha"]

    assert built, "no index_complete log captured"
    entry = built[-1]
    assert "job_id" in entry
    assert entry["job_kind"] == "build"
    assert entry["repo"]
    assert "ref" in entry
