""".rbtrignore file parsing — gitignore semantics.

Each repo can place a `.rbtrignore` at its root to control
which files the index skips. Supports gitignore syntax
including negation (`!pattern`) for force-includes.

Uses the `pathspec` library for matching.
"""

from __future__ import annotations

from pathlib import Path

import pathspec

_DEFAULT_PATTERNS = """\
# Default exclusions
.rbtr/
"""


def parse_ignore(content: str) -> pathspec.PathSpec:
    """Parse ignore patterns from a string (gitignore syntax)."""
    return pathspec.PathSpec.from_lines("gitignore", content.splitlines())


def default_ignore() -> pathspec.PathSpec:
    """Return the default ignore spec when no `.rbtrignore` exists."""
    return parse_ignore(_DEFAULT_PATTERNS)


def load_ignore(repo_root: Path) -> pathspec.PathSpec:
    """Load `.rbtrignore` from *repo_root*, falling back to defaults."""
    ignore_file = repo_root / ".rbtrignore"
    if ignore_file.is_file():
        return parse_ignore(ignore_file.read_text(encoding="utf-8"))
    return default_ignore()
