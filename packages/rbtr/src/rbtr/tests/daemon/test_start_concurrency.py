"""Concurrency regression: racing `daemon start` calls converge.

The bug: a `start_daemon` caller that lost the spawn race waited
for *its own* pid and timed out with "Daemon failed to start
within 5 s" (exit 2).  Several near-simultaneous `daemon start`
invocations must instead all succeed against the single daemon
that wins the DuckDB lock.

This is an inter-process race, so the faithful test launches real
CLI subprocesses against an isolated data dir — no mocking.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rbtr.config import Config
from rbtr.daemon.status import read_status
from rbtr.tests.conftest import run_cli


def test_concurrent_starts_converge_on_one_daemon(isolated_db: Path) -> None:
    runtime_dir = Config(data_dir=Path(os.environ["RBTR_DATA_DIR"])).runtime_dir
    starts = 3
    try:
        with ThreadPoolExecutor(max_workers=starts) as pool:
            results = list(pool.map(lambda _: run_cli(["daemon", "start"]), range(starts)))

        # Every caller succeeds; none reports the lost-race timeout.
        for r in results:
            assert r.returncode == 0, r.stderr
            assert "Daemon failed to start" not in r.stderr

        # Exactly one daemon ended up running, and `status` agrees with
        # the status file (same pid) — the losers reused the winner.
        status = read_status(runtime_dir)
        assert status is not None
        report = json.loads(run_cli(["--json", "daemon", "status"]).stdout)
        assert report["running"] is True
        assert report["pid"] == status.pid
    finally:
        run_cli(["daemon", "stop"])
