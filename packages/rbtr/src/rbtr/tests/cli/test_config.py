"""Tests for `rbtr config`."""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO

import pytest
from rich.console import Console

from rbtr.cli import ConfigCmd


def test_config_json_flag_outputs_valid_json() -> None:
    """`rbtr config --json` emits valid JSON with expected keys."""
    proc = subprocess.run(
        [sys.executable, "-m", "rbtr", "--json", "config"],
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert isinstance(payload["chunk_lines"], int)

    assert payload["db_name"] == "test_index.duckdb"
    assert "runtime_dir" in payload
    assert "db_path" in payload


def test_config_piped_outputs_valid_json() -> None:
    """Piped `rbtr config` falls back to JSON."""

    proc = subprocess.run(
        [sys.executable, "-m", "rbtr", "config"],
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "data_dir" in payload
    assert "db_path" in payload


def test_config_tty_prints_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """When stdout is a tty the command prints a rich table."""
    buf = StringIO()
    monkeypatch.setattr("rbtr.cli._out", Console(file=buf, highlight=False))
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    ConfigCmd().cli_cmd()

    out = buf.getvalue()
    assert "data_dir" in out
    assert "config file:" in out
