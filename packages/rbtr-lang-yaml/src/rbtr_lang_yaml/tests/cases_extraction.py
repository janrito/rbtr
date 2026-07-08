"""YAML extraction test cases (mapping keys -> config_key)."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]


@case(tags=["symbol"])
def case_yaml_top_level_keys() -> SymbolCase:
    """YAML splits by top-level mapping keys."""
    src = """\
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
"""
    return (
        "yaml",
        src,
        [
            ("config_key", "name", ""),
            ("config_key", "on", ""),
            ("config_key", "jobs", ""),
        ],
    )
