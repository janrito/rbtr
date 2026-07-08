"""Load tree-sitter queries from `.scm` files beside each plugin.

Query text lives in `<name>.scm` inside the plugin's own package,
loaded at import time via `load_query` — not embedded inline in Python.
See ARCHITECTURE "Where queries live" for the rationale.
"""

from __future__ import annotations

from importlib.resources import files


def load_query(package: str, name: str) -> str:
    """Return the text of a `<name>.scm` query file in *package*.

    Args:
        package: The plugin package to resolve against, normally the
            caller's own `__package__` (e.g. `"rbtr.languages.python"`).
        name: Query filename without the `.scm` extension (e.g.
            `"python"`, `"injections"`).

    Returns:
        The file's contents verbatim.
    """
    return (files(package) / f"{name}.scm").read_text(encoding="utf-8")
