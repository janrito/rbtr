"""Fixtures for the SQL plugin's tests.

Re-exposes the shared harness fixtures from `rbtr.languages.testkit`
(installed via `rbtr[testkit]`) so pytest discovers them, and defines
this package's `samples/` location.
"""

from __future__ import annotations

from pathlib import Path

from rbtr.languages.testkit import language_manager, snapshot_json

__all__ = ["SAMPLES_DIR", "language_manager", "snapshot_json"]

SAMPLES_DIR = Path(__file__).parent / "samples"
