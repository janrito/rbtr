"""TOML extraction test cases (tables and array-tables -> config_key)."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]


@case(tags=["symbol"])
def case_toml_splits_by_table() -> SymbolCase:
    """TOML splits by tables; a dotted table is named by its last segment."""
    src = """\
[project]
name = "rbtr"

[tool.ruff]
line-length = 99
"""
    return (
        "toml",
        src,
        [
            ("config_key", "project", ""),
            ("config_key", "ruff", "tool"),
        ],
    )
