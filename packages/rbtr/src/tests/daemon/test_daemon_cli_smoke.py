"""End-to-end smoke test for the daemon lifecycle via the CLI.

Catches regressions that only surface when the real subprocess
launch path is exercised (e.g. `python -m rbtr` failing because
`__main__.py` is missing, or `rbtr daemon serve` being hidden
from the subcommand parser).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from rbtr.daemon.status import read_status


@pytest.fixture
def isolated_home(monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    # IPC paths on macOS are capped at 103 chars, so pytest's tmp_path
    # is too deep. Use a short system tempdir and clean up ourselves.
    home = Path(tempfile.mkdtemp(prefix="rbtr"))
    monkeypatch.setenv("RBTR_HOME", str(home))
    try:
        yield home
    finally:
        shutil.rmtree(home, ignore_errors=True)


def _run(
    args: list[str], home: Path, *, timeout: float = 15.0
) -> subprocess.CompletedProcess[str]:
    # Invokes the system under test (the ``rbtr`` CLI).  Not setup
    # — this is the test action, expressed once instead of repeating
    # the same subprocess.run boilerplate at every call site.
    env = {**os.environ, "RBTR_HOME": str(home)}
    return subprocess.run(
        [sys.executable, "-m", "rbtr", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_start_stop_lifecycle(isolated_home: Path) -> None:
    # start
    proc = _run(["daemon", "start"], isolated_home)
    assert proc.returncode == 0, proc.stderr

    status = read_status(isolated_home)
    assert status is not None
    assert status.pid > 0
    assert status.rpc.startswith("ipc://")

    # daemon status reports running with pid + rpc endpoint
    proc = _run(["--json", "daemon", "status"], isolated_home)
    assert proc.returncode == 0
    report = json.loads(proc.stdout)
    assert report["running"] is True
    assert report["pid"] > 0
    assert report["rpc"].startswith("ipc://")

    # stop
    proc = _run(["daemon", "stop"], isolated_home)
    assert proc.returncode == 0, proc.stderr

    # give atexit cleanup a moment to run
    for _ in range(20):
        if read_status(isolated_home) is None:
            break
        time.sleep(0.1)
    assert read_status(isolated_home) is None


def test_start_when_already_running_is_idempotent(isolated_home: Path) -> None:
    try:
        first = _run(["daemon", "start"], isolated_home)
        assert first.returncode == 0

        second = _run(["daemon", "start"], isolated_home)
        assert second.returncode == 0
        assert "already running" in second.stderr.lower()
    finally:
        _run(["daemon", "stop"], isolated_home)
