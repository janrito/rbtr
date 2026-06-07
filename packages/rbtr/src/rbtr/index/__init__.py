"""Code index — structural parsing, storage, and search for code review."""

from __future__ import annotations

import importlib.resources

_SQL_PKG = importlib.resources.files(__name__) / "sql"


def load_sql(name: str) -> str:
    """Read a `.sql` file from this package's `sql/` directory."""
    return (_SQL_PKG / name).read_text(encoding="utf-8")
