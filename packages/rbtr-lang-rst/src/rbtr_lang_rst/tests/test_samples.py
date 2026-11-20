"""reStructuredText sample extraction: the `samples/rst/` project built as rbtr builds a repository and read as an agent reads it.

The project includes a helpers.py the docs cross-reference, so the snapshot captures
the doc sections and the python chunks, and its edges include the rst->python references.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_every_line_in_a_chunk, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "rst") as index:
        symbols = outline(index)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("config.rst", "Config"): {("rst.rst", ":doc:`config`", "documents")},
        ("guide.rst", "Greeting options"): {("rst.rst", "`the guide <guide.rst>`_", "documents")},
        ("guide.rst", "Greetings Table"): {
            ("rst.rst", ":ref:`the greetings table <greetings table>`", "documents"),
            ("rst.rst", "`the guide <guide.rst>`_", "documents"),
        },
        ("guide.rst", "Guide"): {("rst.rst", "`the guide <guide.rst>`_", "documents")},
        ("guide.rst", "Salutations"): {("rst.rst", "`the guide <guide.rst>`_", "documents")},
        ("guide.rst", "format_greeting"): {
            ("rst.rst", ":ref:`format_greeting`", "documents"),
            ("rst.rst", "`the guide <guide.rst>`_", "documents"),
        },
        ("guide.rst", "greeting-options"): {
            ("rst.rst", ":ref:`the options <greeting-options>`", "documents"),
            ("rst.rst", "`the guide <guide.rst>`_", "documents"),
            ("rst.rst", "`the guide's options <guide.rst#greeting-options>`_", "documents"),
        },
        ("guide.rst", "salutations"): {
            ("rst.rst", ":ref:`salutations`", "documents"),
            ("rst.rst", "`the guide <guide.rst>`_", "documents"),
        },
        ("helpers.py", "Greeter"): {("rst.rst", ":class:`Greeter`", "documents")},
        ("helpers.py", "format_greeting"): {("rst.rst", ":func:`format_greeting`", "documents")},
        ("rst.rst", "Usage"): {("rst.rst", "`usage <#usage>`_", "documents")},
        ("topics/detail.rst", "Detail"): {("topics/intro.rst", ":doc:`detail`", "documents")},
        ("topics/intro.rst", "Intro"): {("rst.rst", ":doc:`topics/intro`", "documents")},
    }
    with indexed(Path(__file__).parent / "samples" / "rst") as index:
        assert_reached(index, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "rst") as index:
        assert_every_line_in_a_chunk(index)
