"""Tests for daemon status file management."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest


@pytest.fixture
def status_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestStatusFile:
    def test_status_path(self, status_dir: Path) -> None:
        from rbtr.daemon.status import status_path

        assert status_path(status_dir) == status_dir / "daemon.json"

    def test_write_and_read_roundtrip(self, status_dir: Path) -> None:
        from rbtr.daemon.status import read_status, write_status

        write_status(
            status_dir,
            pid=12345,
            rpc="ipc:///tmp/daemon.rpc",
            pub="ipc:///tmp/daemon.pub",
            version="1.2.3",
        )

        status = read_status(status_dir)
        assert status is not None
        assert status.pid == 12345
        assert status.rpc == "ipc:///tmp/daemon.rpc"
        assert status.pub == "ipc:///tmp/daemon.pub"
        assert status.version == "1.2.3"
        # started_at is set by write_status
        assert status.started_at.endswith("Z")

    def test_read_missing_returns_none(self, status_dir: Path) -> None:
        from rbtr.daemon.status import read_status

        assert read_status(status_dir) is None

    def test_remove_status_deletes_file(self, status_dir: Path) -> None:
        from rbtr.daemon.status import read_status, remove_status, write_status

        write_status(
            status_dir,
            pid=999,
            rpc="ipc:///rpc",
            pub="ipc:///pub",
            version="0.0.1",
        )
        assert read_status(status_dir) is not None

        remove_status(status_dir)
        assert read_status(status_dir) is None

    def test_remove_status_missing_is_no_op(self, status_dir: Path) -> None:
        from rbtr.daemon.status import remove_status

        remove_status(status_dir)  # must not raise

    def test_write_is_atomic(self, status_dir: Path) -> None:
        # Verify the file is not half-written during write
        from rbtr.daemon.status import write_status

        write_status(
            status_dir,
            pid=54321,
            rpc="ipc:///rpc2",
            pub="ipc:///pub2",
            version="9.9.9",
        )

        # Should always be valid JSON
        path = status_dir / "daemon.json"
        data = json.loads(path.read_text())
        assert data["pid"] == 54321


# ── DaemonStatusReport (CLI-only model) ──────────────────────────────


def test_daemon_status_report_running_false_has_only_required_field() -> None:
    from rbtr.daemon.status import DaemonStatusReport

    report = DaemonStatusReport(running=False)
    assert report.running is False
    assert report.pid is None
    assert report.rpc is None
    assert report.pub is None
    assert report.version is None
    assert report.uptime_seconds is None


def test_daemon_status_report_running_true_populated() -> None:
    from rbtr.daemon.status import DaemonStatusReport

    report = DaemonStatusReport(
        running=True,
        pid=12345,
        rpc="ipc:///tmp/daemon.rpc",
        pub="ipc:///tmp/daemon.pub",
        version="1.2.3",
        uptime_seconds=42.5,
    )
    assert report.running is True
    assert report.pid == 12345
    assert report.uptime_seconds == 42.5


def test_uptime_seconds_from_started_at() -> None:
    from datetime import datetime

    from rbtr.daemon.status import uptime_seconds

    # 60 s ago
    started = datetime.now(tz=UTC).replace(microsecond=0).timestamp() - 60
    import time as _time

    iso = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(started))
    assert uptime_seconds(iso) >= 60.0
    assert uptime_seconds(iso) < 62.0


def test_daemon_status_report_forbids_extra_fields() -> None:
    from pydantic import ValidationError

    from rbtr.daemon.status import DaemonStatusReport

    with pytest.raises(ValidationError):
        DaemonStatusReport(running=False, bogus="field")  # type: ignore[call-arg]  # testing strict
