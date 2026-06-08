"""Subprocess helpers for driving rbtr from rbtr-eval.

Every shell-out goes through `run_rbtr`, which invokes
`[sys.executable, "-m", "rbtr", ...]`.  Never `"rbtr"` by
name — that resolves against `$PATH` and could pick up a
different install.  Using this interpreter guarantees the
child is the same rbtr rbtr-eval was built against.

`daemon_session` starts one daemon bound to an isolation
root (data / config / log under that root; cache stays
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
from pathlib import Path

from rbtr.config import Config
from rbtr.daemon.client import DaemonClient


def run_rbtr(
    args: list[str], *, check: bool = True, capture_output: bool = False
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
def daemon_session(
    data_dir: Path,
    config_dir: Path,
    log_dir: Path,
    *,
    recv_timeout_ms: int | None = None,
) -> Iterator[DaemonClient]:
    """Start an isolated daemon; yield a client; stop on exit.

    *data_dir*, *config_dir*, and *log_dir* are the isolation
    directories for the pipeline run.  The cache (embedding
    models) stays on the user's shared platformdirs cache so
    400 MB of weights don't re-download per run.

    The runtime dir (sockets + status file) is derived from
    *data_dir* via `Config(data_dir=data_dir).runtime_dir` and
    lives outside the tree DVC tracks; this is deliberate, so
    a crashed daemon can't leave unhashable sockets in the DVC
    output.

    Stop is best-effort: the daemon exits on ZMQ shutdown
    within about a second, and we don't fail the caller if
    the stop subprocess returns non-zero (the worker may
    already be gone).
    """
    for d in (data_dir, config_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    dir_flags = [
        "--data-dir",
        str(data_dir),
        "--config-dir",
        str(config_dir),
        "--log-dir",
        str(log_dir),
    ]

    # Stop any stale daemon so we always get a fresh process
    # running the current code.  Best-effort: if nothing is
    # running, the stop command exits non-zero harmlessly.
    run_rbtr([*dir_flags, "daemon", "stop"], check=False, capture_output=True)
    run_rbtr([*dir_flags, "daemon", "start"])
    try:
        with DaemonClient(
            runtime_dir=Config(data_dir=data_dir).runtime_dir,
            recv_timeout_ms=recv_timeout_ms,
        ) as client:
            yield client
    finally:
        run_rbtr(
            [*dir_flags, "daemon", "stop"],
            check=False,
            capture_output=True,
        )
