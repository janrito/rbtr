"""Extraction test cases for all languages.

Each `@case` function returns a tuple of test data consumed by
`test_symbol_extraction.py` and `test_import_extraction.py`
via `@parametrize_with_cases`.

Organisation: one section per language, cases tagged by behavior
(`symbol`, `import`, `multi_import`, `mixed`).
"""

from __future__ import annotations

import pytest
from pytest_cases import case

# Return types per tag — each @case function returns one of these.
type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str]]
type MultiImportCase = tuple[str, str, int, list[dict[str, str]]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Multi-import ─────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Multi-import ─────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Multi-import ─────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Multi-import ─────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Multi-import ─────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ═════════════════════��════════════════════════════════════��══════════
# JSON
# ════════════════════════════════════════════════���════════════════════


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
            ("doc_section", "name", ""),
            ("doc_section", "version", ""),
            ("doc_section", "dependencies", ""),
        ],
    )


# ══════════════════════════════════════════════════��══════════════════
# TOML
# ════════════��════════════════════════════════════════════════════════


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
            ("doc_section", "project", ""),
            ("doc_section", "ruff", "tool"),
        ],
    )


# ═��═════════════════════════════════════════��═════════════════════════
# YAML
# ═══════════════════════════════════════════════════════════��═════════


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
            ("doc_section", "name", ""),
            ("doc_section", "on", ""),
            ("doc_section", "jobs", ""),
        ],
    )


# ═════���══════���════════════════════════════════════════════════════════
# HCL
# ═══��═══════════════════════════════════════���═════════════════════════


@case(tags=["symbol"])
def case_hcl_splits_by_blocks() -> SymbolCase:
    """HCL splits by top-level blocks."""
    src = """\
resource "aws_instance" "web" {
  ami = "ami-12345"
}

variable "region" {
  default = "us-east-1"
}
"""
    return (
        "hcl",
        src,
        [
            ("doc_section", "resource aws_instance web", ""),
            ("doc_section", "variable region", ""),
        ],
    )


# ═════════════════════════════════════════════════════════════════════
# Module-level variables (cross-language fan-out)
# ═════════════════════════════════════════════════════════════════════


@case(tags=["symbol"])
def case_bash_assignment() -> SymbolCase:
    """Top-level variable assignment."""
    return "bash", "MAX=100\n", [("variable", "MAX", "")]


# ═════════════════════════════════════════════════════════════════════
# Module-level destructuring & multiple assignment (flat)
# ═════════════════════════════════════════════════════════════════════

_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


# ── Known limitations: nested / chained (strict xfail) ───────────────
