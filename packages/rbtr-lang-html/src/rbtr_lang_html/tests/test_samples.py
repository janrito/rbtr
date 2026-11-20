"""HTML sample extraction: the `samples/html/` project, built as rbtr builds a repository and read as an agent reads it.

The sample links a `.js` and a `.css` and embeds inline `<script>`/`<style>`,
so the javascript and css plugins (dev dependencies) extract the linked and
injected chunks, and its edges include the cross-file links.
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
    with indexed(Path(__file__).parent / "samples" / "html") as index:
        symbols = outline(index)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("greeter.js", "greet"): {("html.html", '<script src="greeter.js"></script>', "imports")},
        ("html.html", "intro"): {
            ("pages/about.html", '<a href="../html.html#intro">the introduction</a>', "imports")
        },
        ("pages/about.html", "contributors"): {
            ("html.html", '<a href="pages/about.html#contributors">contributors</a>', "imports")
        },
        ("pages/about.html", "history"): {
            ("html.html", '<a href="pages/about.html#history">history</a>', "imports")
        },
        ("pages/about.html", "schedule"): {
            ("html.html", '<a href="pages/about.html#schedule">schedule</a>', "imports")
        },
        ("pages/about.html", "team"): {
            ("html.html", '<a href="pages/about.html#team">the team</a>', "imports")
        },
        ("pages/about.js", "describe"): {
            ("pages/about.html", '<script src="about.js"></script>', "imports")
        },
        ("styles.css", ".greeter"): {
            ("html.html", '<link rel="stylesheet" href="styles.css">', "imports")
        },
    }
    with indexed(Path(__file__).parent / "samples" / "html") as index:
        assert_reached(index, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "html") as index:
        assert_every_line_in_a_chunk(index)
