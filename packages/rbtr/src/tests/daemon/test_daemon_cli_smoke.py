"""End-to-end smoke test for the daemon lifecycle via the CLI.

Catches regressions that only surface when the real subprocess
launch path is exercised (e.g. `python -m rbtr` failing because
`__main__.py` is missing, or `rbtr daemon serve` being hidden
from the subcommand parser).
"""

from __future__ import annotations

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
def isolated_user_dir(monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    # IPC paths on macOS are capped at 103 chars, so pytest's tmp_path
    # is too deep. Use a short system tempdir and clean up ourselves.
    user_dir = Path(tempfile.mkdtemp(prefix="rbtr"))
    monkeypatch.setenv("RBTR_USER_DIR", str(user_dir))
    try:
        yield user_dir
    finally:
        shutil.rmtree(user_dir, ignore_errors=True)


def _run(args: list[str], user_dir: Path, *, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "RBTR_USER_DIR": str(user_dir)}
    return subprocess.run(
        [sys.executable, "-m", "rbtr", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_start_stop_lifecycle(isolated_user_dir: Path) -> None:
    # start
    proc = _run(["daemon", "start"], isolated_user_dir)
    assert proc.returncode == 0, proc.stderr

    status = read_status(isolated_user_dir)
    assert status is not None
    assert status.pid > 0
    assert status.rpc.startswith("ipc://")

    # status command reports running
    proc = _run(["--json", "daemon", "status"], isolated_user_dir)
    assert proc.returncode == 0
    assert '"exists":true' in proc.stdout

    # stop
    proc = _run(["daemon", "stop"], isolated_user_dir)
    assert proc.returncode == 0, proc.stderr

    # give atexit cleanup a moment to run
    for _ in range(20):
        if read_status(isolated_user_dir) is None:
            break
        time.sleep(0.1)
    assert read_status(isolated_user_dir) is None


def test_start_when_already_running_is_idempotent(isolated_user_dir: Path) -> None:
    try:
        first = _run(["daemon", "start"], isolated_user_dir)
        assert first.returncode == 0

        second = _run(["daemon", "start"], isolated_user_dir)
        assert second.returncode == 0
        assert "already running" in second.stderr.lower()
    finally:
        _run(["daemon", "stop"], isolated_user_dir)
