"""rbtr — structural code index."""

from __future__ import annotations

import importlib.metadata


def get_version() -> str:
    """Return the installed package version, or ``'dev'``."""
    try:
        return importlib.metadata.version("rbtr")
    except importlib.metadata.PackageNotFoundError:
        return "dev"
