"""Workspace (`.rbtr/`) discovery and path resolution."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pygit2


@cache
def workspace_dir() -> Path:
    """Find the `.rbtr/` directory by walking CWD → git root.

    Nearest existing `.rbtr/` wins. Falls back to `{git_root}/.rbtr`.
    """
    cwd = Path.cwd().resolve()
    git_path = pygit2.discover_repository(str(cwd))
    if git_path is None:
        return cwd / ".rbtr"

    root = Path(git_path).resolve().parent
    current = cwd
    while True:
        candidate = current / ".rbtr"
        if candidate.is_dir():
            return candidate
        if current == root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return root / ".rbtr"


def resolve_path(value: str) -> Path:
    """Resolve a config path against the workspace.

    `${WORKSPACE}/x` expands the placeholder. Absolute paths
    pass through. Relative paths resolve against the project root.
    """
    if "${WORKSPACE}" in value:
        return Path(value.replace("${WORKSPACE}", str(workspace_dir())))
    p = Path(value)
    if p.is_absolute():
        return p
    return workspace_dir().parent / p
