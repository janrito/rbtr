"""Tests for `rbtr config`."""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO

from rbtr.cli import ConfigCmd


def test_config_json_flag_outputs_the_daemon_config_envelope() -> None:
    """`rbtr config --json` emits the config envelope with plugins.

    No daemon runs in the test environment, so this exercises the local
    fallback; the payload shape is identical either way.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "rbtr", "--json", "config"],
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)

    assert payload["kind"] == "daemon_config"
    assert payload["rbtr_version"]

    cfg = payload["config"]
    assert isinstance(cfg["chunk_lines"], int)
    assert cfg["db_name"] == "test_index.duckdb"
    assert "runtime_dir" in cfg
    assert "db_path" in cfg

    python = next(p for p in payload["plugins"] if p["language"] == "python")
    assert python["package"] == "rbtr-lang-python"
    assert python["version"]
    assert isinstance(python["extraction_serial"], int)


def test_config_piped_outputs_valid_json() -> None:
    """Piped `rbtr config` falls back to JSON; config nests under `config`."""

    proc = subprocess.run(
        [sys.executable, "-m", "rbtr", "config"],
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    assert proc.returncode == 0, proc.stderr
    cfg = json.loads(proc.stdout)["config"]
    assert "data_dir" in cfg
    assert "db_path" in cfg


def test_config_tty_prints_config_and_plugin_tables(rendered: StringIO) -> None:
    """A tty gets the settings table, the config-file line, and the plugins table."""
    ConfigCmd().cli_cmd()

    out = rendered.getvalue()
    assert "data_dir" in out
    assert "config file:" in out
    assert "language plugins" in out
    assert "rbtr-lang-python" in out
