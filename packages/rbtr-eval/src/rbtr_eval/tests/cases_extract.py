"""Cases for extract query generation.

Each case returns `(content, language, name, expected_provenances)`
where `expected_provenances` is the set of provenance values the
symbol should produce.  Docstring detection uses tree-sitter
via `extract_doc_spans` — tested through the observable
output (presence or absence of the `docstring` provenance).
"""

from __future__ import annotations

import pytest
from pytest_cases import case

from rbtr.languages import get_manager

_mgr = get_manager()
_skip_no_rust = pytest.mark.skipif(
    _mgr.load_grammar("rust") is None,
    reason="rust tree-sitter grammar not installed",
)
_skip_no_ts = pytest.mark.skipif(
    _mgr.load_grammar("typescript") is None,
    reason="typescript tree-sitter grammar not installed",
)


@case(tags=["yields_queries"])
def case_symbol_with_docstring() -> tuple[str, str, str, set[str]]:
    """Python function with docstring produces name + body + docstring."""
    content = """\
def greet(name):
    \"\"\"Return a friendly greeting for the given person.\"\"\"
    return f"hello {name}"
"""
    return content, "python", "greet", {"name", "body", "docstring"}


@case(tags=["yields_queries"])
def case_symbol_without_docstring() -> tuple[str, str, str, set[str]]:
    """Rust function without doc produces name + body only."""
    content = """\
fn process_items(items: Vec<Item>) -> Result<Vec<Output>> {
    items.iter().map(|i| transform(i)).collect()
}
"""
    return content, "rust", "process_items", {"name", "body"}


@case(tags=["yields_queries"])
def case_symbol_too_short_for_body() -> tuple[str, str, str, set[str]]:
    """Very short function produces name only."""
    content = "fn x() {}"
    return content, "rust", "x", {"name"}


@case(tags=["yields_queries"])
@_skip_no_rust
def case_rust_with_doc_comment() -> tuple[str, str, str, set[str]]:
    """Rust function with /// doc comment produces all three."""
    content = """\
/// Return a friendly greeting for the specified person.
fn greet(name: &str) -> String {
    format!("hello {}", name)
}
"""
    return content, "rust", "greet", {"name", "body", "docstring"}


@case(tags=["yields_queries"])
@_skip_no_ts
def case_ts_with_jsdoc() -> tuple[str, str, str, set[str]]:
    """TypeScript function with JSDoc produces all three."""
    content = """\
/**
 * Return a friendly greeting for the specified person.
 */
function greet(name: string): string {
    return `hello ${name}`;
}
"""
    return content, "typescript", "greet", {"name", "body", "docstring"}


@case(tags=["yields_queries"])
@_skip_no_ts
def case_ts_without_doc() -> tuple[str, str, str, set[str]]:
    """TypeScript function without JSDoc produces name + body only."""
    content = """\
function greet(name: string): string {
    return `hello ${name}`;
}
"""
    return content, "typescript", "greet", {"name", "body"}


@case(tags=["yields_queries"])
def case_prose_with_heading() -> tuple[str, str, str, set[str]]:
    """Markdown section with heading produces name + body."""
    content = """\
# Getting Started

This guide walks you through the installation process.
"""
    return content, "markdown", "Getting Started", {"name", "body"}


@case(tags=["yields_queries"])
def case_prose_no_heading() -> tuple[str, str, str, set[str]]:
    """Headingless paragraph produces body only."""
    content = """\
This guide walks you through the installation process for the project.
"""
    return content, "markdown", "", {"body"}
