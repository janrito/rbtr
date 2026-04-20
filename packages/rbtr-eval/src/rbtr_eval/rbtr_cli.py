"""Subprocess helpers for driving rbtr from rbtr-eval.

Every shell-out goes through `run_rbtr`, which invokes
`[sys.executable, "-m", "rbtr", ...]`.  Never `"rbtr"` by
name — that resolves against `$PATH` and could pick up a
different install.  Using this interpreter guarantees the
child is the same rbtr rbtr-eval was built against.

`daemon_session` starts one daemon bound to an explicit home,
yields a typed `DaemonClient`, and stops the daemon on exit.
The measure and tune stages each open exactly one; every
repo, variant, and query inside the stage shares the same
warm process and loaded embedding model.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

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
def daemon_session(home: Path) -> Iterator[DaemonClient]:
    """Start a daemon bound to *home*; yield a client; stop on exit.

    The client is constructed with `sock_dir=home` explicitly
    so it never picks up the caller's `RBTR_HOME`.  The stop
    is best-effort: the daemon exits on ZMQ shutdown within
    about a second, and we don't fail the caller if the stop
    subprocess returns non-zero (the worker may already be
    gone).
    """
    home.mkdir(parents=True, exist_ok=True)
    run_rbtr(["--home", str(home), "daemon", "start"])
    try:
        with DaemonClient(sock_dir=home) as client:
            yield client
    finally:
        run_rbtr(
            ["--home", str(home), "daemon", "stop"],
            check=False,
            capture_output=True,
        )
