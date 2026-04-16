"""Tests for watched repos persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def watched_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestWatchedPersistence:
    def test_save_empty_watcher_skips_write(self, watched_dir: Path) -> None:
        from rbtr.daemon.watcher import RefWatcher

        watcher = RefWatcher()
        watcher.save(watched_dir)

        # Empty watcher → save is a no-op
        path = watched_dir / "watched.json"
        assert not path.exists()

    def test_save_writes_non_empty_list(self, watched_dir: Path) -> None:
        from rbtr.daemon.watcher import RefWatcher

        watcher = RefWatcher()
        # Manually set a repo path (normally set by register())
        watcher._refs["/some/repo"] = "abc123"
        watcher.save(watched_dir)

        path = watched_dir / "watched.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == ["/some/repo"]

    def test_load_reads_watched_repos(self, watched_dir: Path) -> None:
        from rbtr.daemon.watcher import RefWatcher

        path = watched_dir / "watched.json"
        path.write_text(json.dumps(["/some/repo", "/other/repo"]) + "\n")

        watcher = RefWatcher()
        watcher.load(watched_dir)

        # Non-existent repos are skipped
        assert "/some/repo" not in watcher._refs
        assert "/other/repo" not in watcher._refs

    def test_load_missing_file_is_no_op(self, watched_dir: Path) -> None:
        from rbtr.daemon.watcher import RefWatcher

        watcher = RefWatcher()
        watcher.load(watched_dir)  # no file exists
        assert watcher.repos() == []

    def test_load_corrupt_file_is_no_op(self, watched_dir: Path) -> None:
        from rbtr.daemon.watcher import RefWatcher

        path = watched_dir / "watched.json"
        path.write_text("not valid json{")

        watcher = RefWatcher()
        watcher.load(watched_dir)
        assert watcher.repos() == []
