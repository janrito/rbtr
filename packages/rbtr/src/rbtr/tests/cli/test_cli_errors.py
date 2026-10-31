"""The CLI's error contract under JSON output.

Piped or `--json`, a failing command writes rbtr's `ErrorResponse` to
stdout, like any other response, and keeps its exit code; stderr
carries no error line.  Runs the real CLI in a subprocess, whose stdout
is a pipe, as a caller's would be.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.daemon.messages import ErrorCode, ErrorResponse, IndexState, StatusResponse
from rbtr.errors import ExitCode
from rbtr.tests.conftest import run_cli


@pytest.mark.parametrize(
    ("args", "in_repo", "env", "code", "exit_code"),
    [
        # A `RepoNotFoundError`: the path is not a git repository.
        (["read-symbol", "load_config"], False, {}, ErrorCode.REPO_NOT_FOUND, ExitCode.ERROR),
        # A read with no daemon, before anything was indexed.
        (["read-symbol", "load_config"], True, {}, ErrorCode.INDEX_NOT_BUILT, ExitCode.ERROR),
        # Arguments the command model refuses.
        (["unwatch"], True, {}, ErrorCode.INVALID_REQUEST, ExitCode.ERROR),
        # Too little memory for DuckDB to create the index.
        (
            ["gc"],
            True,
            {"RBTR_DUCKDB_MEMORY_LIMIT": "1MiB"},
            ErrorCode.OUT_OF_MEMORY,
            ExitCode.ERROR,
        ),
    ],
)
def test_error_is_an_error_response_on_stdout(
    args: list[str],
    in_repo: bool,
    env: dict[str, str],
    code: ErrorCode,
    exit_code: ExitCode,
    repo_path: str,
    tmp_path: Path,
    isolated_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    where = repo_path if in_repo else str(tmp_path / "not-a-repo")
    result = run_cli(["--json", *args, "--repo-path", where])

    assert result.returncode == exit_code, result.stderr
    assert ErrorResponse.model_validate_json(result.stdout).code == code
    assert "error:" not in result.stderr


def test_status_after_a_failed_read_reports_no_index(repo_path: str, isolated_db: Path) -> None:
    """A read before any build leaves an empty DB file; status still says no index.

    No daemon runs, so stderr says so, beside the reply on stdout.
    """
    run_cli(["--json", "read-symbol", "load_config", "--repo-path", repo_path])

    result = run_cli(["--json", "status", "--repo-path", repo_path])

    assert result.returncode == 0, result.stdout
    status = StatusResponse.model_validate_json(result.stdout)
    assert status.state is IndexState.NO_INDEX
    assert status.indexed_refs == []
    assert status.repo_path == repo_path
    assert "rbtr is not running" in result.stderr
