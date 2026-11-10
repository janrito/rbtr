"""`docstring`: the documentation a chunk's text carries, as written."""

from __future__ import annotations

import pytest

from rbtr.languages.extract import docstring

PYTHON = '''\
def greet(name):
    """Return a friendly greeting."""
    return name
'''

PROSE = """\
# Greeting

Return a friendly greeting.
"""


@pytest.mark.parametrize(
    ("content", "language", "expected"),
    [
        (PYTHON, "python", '"""Return a friendly greeting."""'),
        (PROSE, "markdown", None),
    ],
    ids=["python-docstring", "no-query"],
)
def test_docstring_is_the_first_documented_symbols_documentation(
    content: str, language: str, expected: str | None
) -> None:
    """A symbol's docstring or the doc comment above it, as written; none when there is none."""
    assert docstring(content, language) == expected
