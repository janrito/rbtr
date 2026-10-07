"""Subprocess helpers for driving rbtr from rbtr-eval.

Every shell-out goes through `run_rbtr`, which invokes
`[sys.executable, "-m", "rbtr", ...]`.  Never `"rbtr"` by
name — that resolves against `$PATH` and could pick up a
different install.  Using this interpreter guarantees the
child is the same rbtr rbtr-eval was built against.

`daemon_session` starts one daemon in the isolation the
environment's `RBTR_*` settings name (cache stays
platformdirs-native and shared across runs), yields a typed
`DaemonClient`, and stops the daemon on exit.  Measure and
tune each open exactly one; every repo and query
inside the stage shares the same warm process and loaded
embedding model.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from rbtr.daemon.client import DaemonClient


def run_rbtr(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    """Invoke `python -m rbtr <args>` against this interpreter.

    Never shells out by bare name; the child is always the
    rbtr installed in this interpreter's environment, which
    is the same one `rbtr_eval` imports from.
    """
    return subprocess.run(  # noqa: S603 - trusted args: sys.executable + literal module
        [sys.executable, "-m", "rbtr", *args],
        check=check,
        capture_output=capture_output,
    )


@contextmanager
def daemon_session() -> Iterator[DaemonClient]:
    """Start an isolated daemon; yield a client; stop on exit.

    The daemon runs with rbtr's settings, which the pipeline sets in
    the environment: `RBTR_DATA_DIR`, `RBTR_LOG_DIR` and
    `RBTR_CONFIG_DIR` isolate it.  The cache (embedding models) stays on
    the user's shared platformdirs cache so 400 MB of weights don't
    re-download per run.  The client waits as long as
    `RBTR_DAEMON_WAIT_BUDGET_S` allows.

    The runtime dir (sockets + status file) is derived from the data
    dir and lives outside the tree DVC tracks; this is deliberate, so
    a crashed daemon can't leave unhashable sockets in the DVC output.

    Stop is best-effort: the daemon exits on ZMQ shutdown
    within about a second, and we don't fail the caller if
    the stop subprocess returns non-zero (the worker may
    already be gone).
    """
    # Stop any stale daemon so we always get a fresh process
    # running the current code.  Best-effort: if nothing is
    # running, the stop command exits non-zero harmlessly.
    run_rbtr(["daemon", "stop"], check=False, capture_output=True)
    run_rbtr(["daemon", "start"])
    try:
        with DaemonClient() as client:
            yield client
    finally:
        run_rbtr(["daemon", "stop"], check=False, capture_output=True)
