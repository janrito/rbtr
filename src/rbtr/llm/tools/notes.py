"""Edit tool — create and edit files matching ``editable_include`` globs."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic_ai import RunContext

from rbtr.config import config
from rbtr.git.filters import _matches_globs
from rbtr.llm.deps import AgentDeps
from rbtr.llm.tools.common import workspace_toolset


def _is_editable(path: str) -> bool:
    """Check whether *path* is writable by the ``edit`` tool.

    A path is editable when it matches any pattern in
    ``tools.editable_include``.
    """
    return _matches_globs(path, config.tools.editable_include)


@workspace_toolset.tool
def edit(
    ctx: RunContext[AgentDeps],
    path: str,
    new_text: str,
    old_text: str = "",
) -> str:
    """Edit or create a file.

    Args:
        path: File path relative to the repo root
            (e.g. `.rbtr/notes/plan.md`, `.rbtr/AGENTS.md`).
            Must match an `editable_include` pattern.
        new_text: Content to write or insert.
        old_text: Exact text to find and replace.  Empty string
            (default) creates the file or appends to it.
    """
    # Validate path.
    p = PurePosixPath(path)
    if ".." in p.parts:
        return f"Path '{path}' contains '..' — not allowed."
    if not _is_editable(path):
        return f"Path '{path}' is not in `editable_include` — cannot edit."

    resolved = Path(p)

    if not old_text:
        # Create or append.
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if resolved.exists():
            existing = resolved.read_text()
            resolved.write_text(existing + new_text)
            return f"Appended to {path}."
        resolved.write_text(new_text)
        return f"Created {path}."

    # Replace exact match.
    if not resolved.exists():
        return f"File '{path}' does not exist — cannot replace."
    content = resolved.read_text()
    count = content.count(old_text)
    if count == 0:
        return f"old_text not found in '{path}'."
    if count > 1:
        return f"old_text matches {count} times in '{path}' — must be unique."
    resolved.write_text(content.replace(old_text, new_text, 1))
    return f"Replaced in {path}."
