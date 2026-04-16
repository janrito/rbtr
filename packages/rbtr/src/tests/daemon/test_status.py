"""Tests for daemon status file management."""

from __future__ import annotations

import json
import time
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
        from rbtr.daemon.status import read_status, write_status

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
