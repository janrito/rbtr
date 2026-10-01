"""The CLI's error contract under JSON output.

Piped or `--json`, a failing command writes rbtr's `ErrorResponse` to
stdout, like any other response, and keeps its exit code; stderr
carries no error line.  Runs the real CLI in a subprocess, whose stdout
is a pipe, as a caller's would be.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.daemon.messages import ErrorCode, ErrorResponse, StatusResponse
from rbtr.errors import ExitCode
from rbtr.tests.conftest import run_cli


@pytest.mark.parametrize(
    ("args", "in_repo", "code", "exit_code"),
    [
        # An `RbtrError`: the path is not a git repository.
        (["read-symbol", "load_config"], False, ErrorCode.INTERNAL, ExitCode.ERROR),
        # A read with no daemon, before anything was indexed.
        (["read-symbol", "load_config"], True, ErrorCode.INDEX_NOT_BUILT, ExitCode.ERROR),
        # Arguments the command model refuses.
        (["unwatch"], True, ErrorCode.INVALID_REQUEST, ExitCode.ERROR),
    ],
)
def test_error_is_an_error_response_on_stdout(
    args: list[str],
    in_repo: bool,
    code: ErrorCode,
    exit_code: ExitCode,
    repo_path: str,
    tmp_path: Path,
    isolated_db: Path,
) -> None:
    where = repo_path if in_repo else str(tmp_path / "not-a-repo")
    result = run_cli(["--json", *args, "--repo-path", where])

    assert result.returncode == exit_code, result.stderr
    assert ErrorResponse.model_validate_json(result.stdout).code == code
    assert "error:" not in result.stderr


def test_status_after_a_failed_read_reports_no_index(repo_path: str, isolated_db: Path) -> None:
    """A read before any build leaves an empty DB file; status still says no index."""
    run_cli(["--json", "read-symbol", "load_config", "--repo-path", repo_path])

    result = run_cli(["--json", "status", "--repo-path", repo_path])

    assert result.returncode == 0, result.stdout
    assert StatusResponse.model_validate_json(result.stdout).indexed_refs == []
