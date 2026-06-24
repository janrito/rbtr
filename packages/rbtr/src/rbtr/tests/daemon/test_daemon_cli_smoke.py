"""End-to-end smoke test for the daemon lifecycle via the CLI.

Catches regressions that only surface when the real subprocess
launch path is exercised (e.g. `python -m rbtr` failing because
`__main__.py` is missing, or `rbtr daemon serve` being hidden
from the subcommand parser).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from rbtr.config import Config
from rbtr.daemon.status import read_status
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
