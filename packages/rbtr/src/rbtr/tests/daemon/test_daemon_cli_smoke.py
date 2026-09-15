"""End-to-end smoke test for the daemon lifecycle via the CLI.

Catches regressions that only surface when the real subprocess
launch path is exercised (e.g. `python -m rbtr` failing because
`__main__.py` is missing, or `rbtr daemon serve` being hidden
from the subcommand parser).
"""

from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from rbtr.config import Config
from rbtr.daemon.status import is_pid_alive, read_status
from rbtr.index.store import IndexStore
from rbtr.tests.conftest import run_cli


def test_start_stop_lifecycle(isolated_db: Path) -> None:
    runtime_dir = Config(data_dir=Path(os.environ["RBTR_DATA_DIR"])).runtime_dir

    # start
    proc = run_cli(["daemon", "start"])
    assert proc.returncode == 0, proc.stderr

    status = read_status(runtime_dir)
    assert status is not None
    assert status.pid > 0
    assert status.rpc.startswith("ipc://")

    # daemon status reports running with pid + rpc endpoint
    proc = run_cli(["--json", "daemon", "status"])
    assert proc.returncode == 0
    report = json.loads(proc.stdout)
    assert report["running"] is True
    assert report["pid"] > 0
    assert report["rpc"].startswith("ipc://")

    # stop
    proc = run_cli(["daemon", "stop"])
    assert proc.returncode == 0, proc.stderr

    # give atexit cleanup a moment to run
    for _ in range(20):
        if read_status(runtime_dir) is None:
            break
        time.sleep(0.1)
    assert read_status(runtime_dir) is None


def test_status_when_not_running() -> None:
    proc = run_cli(["--json", "daemon", "status"])
    assert proc.returncode == 0
    report = json.loads(proc.stdout)
    assert report["running"] is False
    assert report["pid"] is None


def test_start_when_already_running_is_idempotent(isolated_db: Path) -> None:
    try:
        first = run_cli(["daemon", "start"])
        assert first.returncode == 0

        second = run_cli(["daemon", "start"])
        assert second.returncode == 0
        assert "already running" in second.stderr.lower()
    finally:
        run_cli(["daemon", "stop"])


@pytest.fixture(scope="module")
def orphan_pids() -> Generator[list[int]]:
    """Collect daemon pids that `isolated_db` teardown must have killed.

    Module-scoped, so it is set up before the function-scoped
    `isolated_db` and finalised *after* it -- which is what lets it
    observe whether that teardown did its job.
    """
    pids: list[int] = []
    yield pids
    survivors = [pid for pid in pids if is_pid_alive(pid)]
    for pid in survivors:  # don't leak them just because the check failed
        os.kill(pid, signal.SIGKILL)
    assert not survivors, f"daemons survived isolated_db teardown: {survivors}"


def test_daemon_left_running_is_killed_by_fixture_teardown(
    isolated_db: Path,
    orphan_pids: list[int],
) -> None:
    """A daemon still running at test end must not outlive the run.

    The tests here stop their daemon in a `finally`, which does not run
    when a run is interrupted part-way through a test; the orphan then
    idles indefinitely against a `tmp_path` data dir no later run will
    reuse.  This one deliberately omits `daemon stop` to stand in for
    that, and `orphan_pids` asserts the teardown killed it.
    """
    assert run_cli(["daemon", "start"]).returncode == 0

    status = read_status(Config(data_dir=isolated_db).runtime_dir)
    assert status is not None, "daemon did not record a status file"
    assert is_pid_alive(status.pid), "daemon is not running"

    orphan_pids.append(status.pid)


def test_start_with_db_lock_held_exits_cleanly(isolated_db: Path) -> None:
    """A start that cannot acquire the DuckDB lock fails honestly.

    Holding the exclusive lock in-process makes the spawned
    `daemon serve` die on the lock, so `start_daemon` raises
    `RbtrError`.  The command must catch it and exit 1 (its own
    handler) — not let it escape to the global handler as exit 2,
    which is what the pre-fix `except RuntimeError` did.
    """
    store = IndexStore.from_config(writable=True)  # take the exclusive lock
    try:
        result = run_cli(["daemon", "start"])
        assert result.returncode == 1, result.stderr
        assert "Daemon failed to start" in result.stderr
    finally:
        store.close()
        run_cli(["daemon", "stop"])
