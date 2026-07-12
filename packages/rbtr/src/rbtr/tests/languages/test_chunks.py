"""Tests for plaintext/prose chunking (`rbtr.languages.chunks`)."""

from __future__ import annotations

import pytest

from rbtr.languages.chunks import detect_prose_format


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            """\
Title
=====

Body text.
""",
            "rst",
            id="rst-underline",
        ),
        pytest.param(
            """\
Some text.

.. note::

    A note.
""",
            "rst",
            id="rst-directive",
        ),
        pytest.param(
            """\
# Title

Body text.
""",
            "markdown",
            id="md-atx",
        ),
        pytest.param(
            """\
Just some plain text.
Nothing special about it.
""",
            None,
            id="neither",
        ),
    ],
)
def test_detect_prose_format(content: str, expected: str | None) -> None:
    assert detect_prose_format(content) == expected
