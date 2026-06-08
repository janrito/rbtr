"""PID file management for the daemon."""

from __future__ import annotations

import os
from pathlib import Path


def write_pid(path: Path, pid: int) -> None:
    """Write *pid* to *path*, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid))


def read_pid(path: Path) -> int | None:
    """Read the PID from *path*, or `None` if missing/corrupt."""
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def is_pid_alive(pid: int) -> bool:
    """Check whether a process with *pid* exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we can't signal it
    return True


def is_daemon_running(pid_path: Path) -> bool:
    """Check whether the PID file points to a live process."""
    pid = read_pid(pid_path)
    if pid is None:
        return False
    return is_pid_alive(pid)


def clean_pid(path: Path) -> None:
    """Remove the PID file if it exists."""
    path.unlink(missing_ok=True)
