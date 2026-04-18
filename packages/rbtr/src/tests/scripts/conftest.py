"""Shared fixtures for bench-script tests.

`packages/rbtr/scripts/` is not on `sys.path` by default (the
scripts are meant to be run via `uv run`).  Tests that import
from them add the scripts directory to `sys.path` once at
module collection time.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
