"""Handler for /reload — confirm prompt sources are loaded."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.config import config
from rbtr.constants import RBTR_DIR

if TYPE_CHECKING:
    from .core import Engine


def cmd_reload(engine: Engine) -> None:
    """Report which prompt source files are active.

    Templates are re-read from disk on every LLM turn (no
    caching), so `/reload` is a visibility command — it
    confirms what the next turn will use.
    """
    lines: list[str] = []

    # System prompt source.
    override_name = config.system_prompt_override
    if override_name:
        override = RBTR_DIR / override_name
        if override.is_file():
            lines.append(f"  system:  {override} (override)")
        else:
            lines.append("  system:  built-in")
    else:
        lines.append("  system:  built-in")

    # Append-system file.
    append_name = config.append_system
    if append_name:
        append = RBTR_DIR / append_name
        if append.is_file():
            lines.append(f"  append:  {append}")

    # Project instruction files.
    found: list[str] = []
    missing: list[str] = []
    for name in config.project_instructions:
        if Path(name).is_file():
            found.append(name)
        else:
            missing.append(name)

    if found:
        for name in found:
            lines.append(f"  project: {name}")
    if missing:
        for name in missing:
            lines.append(f"  project: {name} (not found)")

    if not found and not missing:
        lines.append("  project: (none configured)")

    engine._out("Prompt sources:")
    for line in lines:
        engine._out(line)
    engine._out("Instructions will refresh on the next turn.")
