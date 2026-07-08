"""JSON extraction test cases (object keys -> config_key)."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]


@case(tags=["symbol"])
def case_json_top_level_keys() -> SymbolCase:
    """JSON splits by top-level keys."""
    src = """\
{
  "name": "my-project",
  "version": "1.0.0",
  "dependencies": {
    "foo": "^1.0"
  }
}
"""
    return (
        "json",
        src,
        [
            ("config_key", "name", ""),
            ("config_key", "version", ""),
            ("config_key", "dependencies", ""),
        ],
    )
