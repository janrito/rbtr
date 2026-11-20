"""Markdown sample extraction: the `samples/markdown/` project, built as rbtr builds a repository and read as an agent reads it.

The sample embeds fenced code (python, sh, yaml, toml, html, svelte), delegated to
those plugins, and links between files, so the snapshot captures the doc sections
and injected chunks, and its edges include the cross-file links.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "markdown") as build:
        symbols = outline(build)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("api.md", "Format a greeting — the helper"): {
            ("markdown.md", "api.md#format-a-greeting-the-helper", "documents")
        },
        ("config.md", "Configuration"): {("markdown.md", "config.md", "documents")},
        ("config.md", "Locale options {#locale}"): {("markdown.md", "config.md", "documents")},
        ("config.md", "install-notes"): {
            ("markdown.md", "config.md", "documents"),
            ("markdown.md", "config.md#install-notes", "documents"),
        },
        ("config.md", "legacy-options"): {
            ("markdown.md", "config.md", "documents"),
            ("markdown.md", "config.md#legacy-options", "documents"),
        },
        ("guides/setup.md", "Setup"): {("markdown.md", "guides/setup.md", "documents")},
        ("guides/troubleshooting.md", "A step failed"): {
            ("guides/setup.md", "troubleshooting.md", "documents")
        },
        ("guides/troubleshooting.md", "Troubleshooting"): {
            ("guides/setup.md", "troubleshooting.md", "documents")
        },
        ("locales.md", "Locales"): {("markdown.md", "locales.md", "documents")},
        ("markdown.md", "Usage"): {("markdown.md", "#usage", "documents")},
    }
    with indexed(Path(__file__).parent / "samples" / "markdown") as build:
        assert_reached(build, reached)
