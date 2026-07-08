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
