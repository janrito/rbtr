"""The `--scope` CLI flag: case-insensitive, and `all` needs no repository.

Humans may type `all`, `ALL`, or `All`; our `_normalise_scope`
validator folds the case before the enum coerces.  Exercised
through the real CLI subprocess (read-only `status`, so no
write-lock on the shared test store).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..conftest import run_cli


@pytest.mark.parametrize("text", ["all", "ALL", "All", "workspace", "WORKSPACE"])
def test_scope_flag_accepts_any_case(text: str, isolated_db: Path) -> None:
    r = run_cli(["--json", "status", "--scope", text])
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("command", [["status"], ["search", "x"], ["gc"], ["unwatch", "--stale"]])
def test_scope_all_runs_outside_a_git_repository(
    command: list[str], tmp_path: Path, isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every repo is covered from anywhere: the current directory is not read as a repo."""
    monkeypatch.chdir(tmp_path)
    r = run_cli(["--json", *command, "--scope", "all"])
    assert "Not a git repository" not in r.stdout + r.stderr
