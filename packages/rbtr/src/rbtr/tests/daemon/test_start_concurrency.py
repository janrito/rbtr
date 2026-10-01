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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rbtr.config import Config
from rbtr.daemon.status import read_status
from rbtr.index.store import IndexStore
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


def test_a_start_that_loses_the_lock_waits_for_the_holder_to_serve(
    isolated_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The holder stands in for a racing daemon that took the index lock
    # and has not yet written its status file.  The waiting start's own
    # `serve` loses the lock; only then does a daemon come up, however
    # long after that the holder takes to serve.
    log = tmp_path / "logs" / "daemon.log"
    monkeypatch.setenv("RBTR_LOG_DIR", str(log.parent))
    holder = IndexStore.from_config(writable=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(run_cli, ["daemon", "start"])
            deadline = time.monotonic() + 30
            while not (log.exists() and "duckdb_lock_conflict" in log.read_text()):
                assert time.monotonic() < deadline, "the waiting start's serve never met the lock"
                time.sleep(0.1)
            time.sleep(1)  # well past the moment a start used to give up
            holder.close()
            assert run_cli(["daemon", "start"]).returncode == 0

            result = waiting.result()
        assert result.returncode == 0, result.stderr
    finally:
        holder.close()
        run_cli(["daemon", "stop"])
