"""CSS extraction test cases.

Each `@case` returns test data consumed by `test_extraction.py` via
`pytest-cases`. See the plugin docstring for the full source→chunk mapping.
"""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]


@case(tags=["symbol"])
def case_css_rule_sets() -> SymbolCase:
    """CSS splits by rule sets."""
    src = """\
body {
  color: #333;
}

.header {
  background: blue;
}
"""
    return (
        "css",
        src,
        [
            ("class", "body", ""),
            ("class", ".header", ""),
        ],
    )
