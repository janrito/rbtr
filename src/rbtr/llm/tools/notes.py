"""Review notes tool — create and edit files in the notes directory."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic_ai import RunContext

from rbtr.config import config
from rbtr.llm.agent import AgentDeps, agent


@agent.tool
def edit(
    ctx: RunContext[AgentDeps],
    path: str,
    new_text: str,
    old_text: str = "",
) -> str:
    """Edit or create a review notes file.

    Args:
        path: File path relative to the repo root
            (e.g. `.rbtr/notes/plan.md`).  Must be inside the
            notes directory.
        new_text: Content to write or insert.
        old_text: Exact text to find and replace.  Empty string
            (default) creates the file or appends to it.
    """
    notes_dir = Path(config.tools.notes_dir)
    # Validate path.
    p = PurePosixPath(path)
    if ".." in p.parts:
        return f"Path '{path}' contains '..' — not allowed."
    resolved = Path(p)
    try:
        resolved.resolve().relative_to(notes_dir.resolve())
    except ValueError:
        return f"Path must be inside {notes_dir}/ — got '{path}'."

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
