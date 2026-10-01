"""A caller that starts the daemon and later stops it, in one process."""

from __future__ import annotations

import time
from pathlib import Path

from rbtr.daemon.client import start_daemon, stop_daemon


def test_a_daemon_this_process_started_stops_within_the_timeout(isolated_db: Path) -> None:
    # The daemon is this process's child, so once it exits it lingers as a
    # zombie until reaped; a stop that took the zombie for a live daemon
    # waited out the timeout and then reported that it did not stop.
    start_daemon()
    started = time.monotonic()
    stop_daemon(timeout=10.0)
    assert time.monotonic() - started < 10.0
