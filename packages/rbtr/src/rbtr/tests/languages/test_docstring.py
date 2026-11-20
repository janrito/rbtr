"""`docstring`: the documentation a chunk's text carries, as written."""

from __future__ import annotations

import pytest

from rbtr.languages.extract import docstring

PYTHON = '''\
def greet(name):
    """Return a friendly greeting."""
    return name
'''

RUST = """\
/// Return a friendly greeting.
fn greet(name: &str) -> String {
    name.to_string()
}
"""

TYPESCRIPT = """\
/**
 * Return a friendly greeting.
 */
function greet(name: string): string {
    return name;
}
"""

UNDOCUMENTED = """\
fn greet(name: &str) -> String {
    name.to_string()
}
"""

PROSE = """\
# Greeting

Return a friendly greeting.
"""


@pytest.mark.parametrize(
    ("content", "language", "expected"),
    [
        (PYTHON, "python", '"""Return a friendly greeting."""'),
        (RUST, "rust", "/// Return a friendly greeting.\n"),
        (TYPESCRIPT, "typescript", "/**\n * Return a friendly greeting.\n */"),
        (UNDOCUMENTED, "rust", None),
        (PROSE, "markdown", None),
    ],
    ids=["python-docstring", "rust-doc-comment", "typescript-jsdoc", "undocumented", "no-query"],
)
def test_docstring_is_the_first_documented_symbols_documentation(
    content: str, language: str, expected: str | None
) -> None:
    """A symbol's docstring or the doc comment above it, as written; none when there is none."""
    assert docstring(content, language) == expected
