"""Shared fixtures for the in-core language tests.

The reusable harness (fixtures, sample loaders, extraction entry point,
snapshot serialiser) now lives in `rbtr.languages.testkit` — the shipped,
public test-support module that plugin packages import. This conftest
re-exposes the fixtures so pytest discovers them for the in-core tests,
exactly as a plugin package's own conftest does, and defines the core
tests' `samples/` location.
"""

from __future__ import annotations

from pathlib import Path

from rbtr.languages.testkit import language_manager, snapshot_json

__all__ = ["SAMPLES_DIR", "language_manager", "snapshot_json"]

SAMPLES_DIR = Path(__file__).parent / "samples"
