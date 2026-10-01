"""The CLI's error contract under JSON output.

Piped or `--json`, a failing command writes rbtr's `ErrorResponse` to
stdout, like any other response, and keeps its exit code; stderr
carries no error line.  Runs the real CLI in a subprocess, whose stdout
is a pipe, as a caller's would be.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.daemon.messages import ErrorCode, ErrorResponse
from rbtr.errors import ExitCode
from rbtr.tests.conftest import run_cli


@pytest.mark.parametrize(
    ("args", "code", "exit_code"),
    [
        # An `RbtrError`: the path is not a git repository.
        (["read-symbol", "load_config"], ErrorCode.INTERNAL, ExitCode.ERROR),
        # Arguments the command model refuses.
        (["unwatch"], ErrorCode.INVALID_REQUEST, ExitCode.ERROR),
    ],
)
def test_error_is_an_error_response_on_stdout(
    args: list[str], code: ErrorCode, exit_code: ExitCode, tmp_path: Path, isolated_db: Path
) -> None:
    result = run_cli(["--json", *args, "--repo-path", str(tmp_path)])

    assert result.returncode == exit_code, result.stderr
    assert ErrorResponse.model_validate_json(result.stdout).code == code
    assert "error:" not in result.stderr
