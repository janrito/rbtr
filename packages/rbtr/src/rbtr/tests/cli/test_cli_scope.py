"""The `--scope` CLI flag is case-insensitive.

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
