"""Request and job correlation through real daemon operations."""

from __future__ import annotations

import time
from pathlib import Path

import structlog

from rbtr.config import config
from rbtr.daemon.client import DaemonClient
from rbtr.daemon.messages import DaemonConfigRequest, StatusRequest, WatchRequest
from rbtr.daemon.server import DaemonServer
from rbtr.index.index import Index

from .conftest import serving


def test_requests_carry_unique_log_contexts(
    running_daemon: DaemonServer,
    fake_repo: str,
    fake_repo_subdir: str,
    log_output: structlog.testing.LogCapture,
) -> None:
    with DaemonClient(running_daemon.runtime_dir) as client:
        client.send(WatchRequest(repo_path=fake_repo, refs=["HEAD"]))
        client.send(WatchRequest(repo_path=fake_repo_subdir, refs=["HEAD"]))

    written = [e for e in log_output.entries if e["event"] == "watched_refs_added"]
    assert len(written) == 2
    assert all(e["kind"] == "watch" and e["repo"] == fake_repo for e in written)
    assert written[0]["request_id"] != written[1]["request_id"]
    completed = [e for e in log_output.entries if e["event"] == "request_complete"]
    assert len(completed) == 2
    assert all(e["repo"] == fake_repo for e in completed)


def test_binding_is_scoped_to_the_request(
    running_daemon: DaemonServer,
    fake_repo: str,
    log_output: structlog.testing.LogCapture,
) -> None:
    with DaemonClient(running_daemon.runtime_dir) as client:
        for _ in range(config.max_concurrent_reads):
            client.send(StatusRequest(repo_path=fake_repo))
        for _ in range(config.max_concurrent_reads):
            client.send(DaemonConfigRequest())

    configured = [
        e
        for e in log_output.entries
        if e["event"] == "request_complete" and e["kind"] == "daemon_config"
    ]
    assert len(configured) == config.max_concurrent_reads
    assert all("repo" not in e for e in configured)


def test_request_complete_carries_elapsed_ms(
    running_daemon: DaemonServer,
    fake_repo: str,
    log_output: structlog.testing.LogCapture,
) -> None:
    with DaemonClient(running_daemon.runtime_dir) as client:
        client.send(StatusRequest(repo_path=fake_repo))

    done = [e for e in log_output.entries if e["event"] == "request_complete"]
    assert len(done) == 1
    assert isinstance(done[0]["elapsed_ms"], float)
    assert done[0]["elapsed_ms"] >= 0


def test_embed_job_logs_are_correlated(
    running_daemon: DaemonServer,
    log_output: structlog.testing.LogCapture,
) -> None:
    """End-to-end: a background embed job tags its logs with job context.

    No bespoke daemon needed: `seeded_index` holds indexed-but-unembedded
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
    registered_index: Index,
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
        index=registered_index,
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
