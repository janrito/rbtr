"""Daemon status file management.

The status file (`~/.rbtr/daemon.json`) is the single source of
truth for the running daemon's identity. It is written by the
server after binding sockets, and read by clients to discover
the RPC and PUB endpoints.

Format::

    {
      "pid": 12345,
      "rpc": "ipc:///Users/user/.rbtr/daemon.rpc",
      "pub": "ipc:///Users/user/.rbtr/daemon.pub",
      "started_at": "2026-04-16T10:00:00Z",
      "version": "1.2.3"
    }

The server atomically writes the file after binding sockets so
that a client never sees a file with stale endpoints. The client
reads it to find the socket paths, bypassing socket discovery.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict


@dataclass
class DaemonStatus:
    """Contents of the daemon status file."""

    pid: int
    rpc: str
    pub: str
    started_at: str
    version: str


class DaemonStatusReport(BaseModel):
    """CLI output for ``rbtr daemon status``.

    Answers the *process* question — is the daemon running,
    what's its PID, uptime, ZMQ endpoints — and is **not** a
    ZMQ protocol response.  Derived entirely from the on-disk
    status file plus a liveness check on the PID.
    """

    model_config = ConfigDict(extra="forbid")

    running: bool
    pid: int | None = None
    rpc: str | None = None
    pub: str | None = None
    version: str | None = None
    uptime_seconds: float | None = None


def status_path(home: Path) -> Path:
    """Path to the daemon status file."""
    return home / "daemon.json"


def read_status(home: Path) -> DaemonStatus | None:
    """Read the daemon status file, or ``None`` if missing."""
    path = status_path(home)
    try:
        with open(path) as f:
            data = json.load(f)
        return DaemonStatus(
            pid=data["pid"],
            rpc=data["rpc"],
            pub=data["pub"],
            started_at=data["started_at"],
            version=data["version"],
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def write_status(
    home: Path,
    *,
    pid: int,
    rpc: str,
    pub: str,
    version: str,
) -> None:
    """Write the daemon status file atomically.

    Writes to a temp file in the same directory, then renames
    to the final name. This avoids a reader ever seeing a
    partial/truncated file.
    """
    path = status_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".daemon.json.tmp")
    data = {
        "pid": pid,
        "rpc": rpc,
        "pub": pub,
        "started_at": _iso_now(),
        "version": version,
    }
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.rename(path)


def remove_status(home: Path) -> None:
    """Remove the status file if it exists."""
    status_path(home).unlink(missing_ok=True)


def _iso_now() -> str:
    """Current UTC time as ISO 8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def uptime_seconds(started_at: str) -> float:
    """Seconds elapsed since *started_at* (ISO 8601 UTC, Z-suffixed)."""
    started = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc,
    )
    return (datetime.now(tz=timezone.utc) - started).total_seconds()
