"""Prompt rendering — loads markdown templates and fills placeholders.

The system prompt is the only LLM-facing template.  Project-level
instructions (``AGENTS.md`` etc.) and user-level extensions
(``APPEND_SYSTEM.md``) are injected as plain text via template
variables — they are not themselves templates.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import minijinja

from rbtr.config import config
from rbtr.constants import RBTR_DIR
from rbtr.models import BranchTarget, PRTarget

if TYPE_CHECKING:
    from rbtr.state import EngineState

log = logging.getLogger(__name__)


def _load_template(name: str) -> str:
    """Read a .md template from the prompts package."""
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")  # type: ignore[union-attr]  # Traversable always has read_text when joinpath succeeds


def _build_env() -> minijinja.Environment:
    """Create a MiniJinja environment with all prompt templates."""
    env = minijinja.Environment()
    env.add_template("system", _load_template("system.md"))
    env.add_template("index_status", _load_template("index_status.md"))
    env.add_template("compact", _load_template("compact.md"))
    return env


def _read_optional(path: Path) -> str:
    """Read a file if it exists, return empty string otherwise."""
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            log.warning("Could not read %s", path)
    return ""


def _load_project_instructions() -> str:
    """Read and concatenate project instruction files from the repo root.

    File names come from ``config.project_instructions`` (a list,
    default ``["AGENTS.md"]``).  Missing files are silently skipped.
    """
    parts: list[str] = []
    for name in config.project_instructions:
        content = _read_optional(Path(name))
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def _load_append_system() -> str:
    """Read ``APPEND_SYSTEM.md`` from ``~/.config/rbtr/`` if present."""
    return _read_optional(RBTR_DIR / "APPEND_SYSTEM.md")


def _load_system_override() -> str:
    """Read ``SYSTEM.md`` from ``~/.config/rbtr/`` if present."""
    return _read_optional(RBTR_DIR / "SYSTEM.md")


def _context(state: EngineState) -> dict[str, Any]:
    """Build template context from live state."""
    ctx: dict[str, Any] = {
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "owner": state.owner or "unknown",
        "repo": state.repo_name or "unknown",
        "target_kind": "none",
        "base_branch": "",
        "branch": "",
        "pr_number": 0,
        "pr_title": "",
        "pr_author": "",
        "pr_body": "",
        "project_instructions": _load_project_instructions(),
        "append_system": _load_append_system(),
        # Tool config metadata — helps the LLM understand output limits.
        "editable_globs": config.tools.editable_include,
        "max_lines": config.tools.max_lines,
        "max_results": config.tools.max_results,
        "max_grep_hits": config.tools.max_grep_hits,
        "max_requests_per_turn": config.tools.max_requests_per_turn,
    }

    match state.review_target:
        case PRTarget(
            number=n,
            title=t,
            base_branch=base,
            head_branch=head,
            author=a,
            body=b,
        ):
            ctx |= {
                "target_kind": "pr",
                "base_branch": base,
                "branch": head,
                "pr_number": n,
                "pr_title": t,
                "pr_author": a,
                "pr_body": b,
            }
        case BranchTarget(base_branch=base, head_branch=head):
            ctx |= {
                "target_kind": "branch",
                "base_branch": base,
                "branch": head,
            }

    return ctx


def render_system(state: EngineState) -> str:
    """Render the system prompt with live state data.

    If ``~/.config/rbtr/SYSTEM.md`` exists, it replaces the
    built-in template (same Jinja variables are available).
    Otherwise the built-in ``system.md`` is used.
    """
    env = minijinja.Environment()
    override = _load_system_override()
    if override:
        env.add_template("system", override)
    else:
        env.add_template("system", _load_template("system.md"))
    return env.render_template("system", **_context(state))


def render_index_status(*, status: str, tool_names: list[str]) -> str:
    """Render the index status instruction.

    Args:
        status: ``"ready"``, ``"building"``, or ``""`` (no review target).
        tool_names: Names of tools that require the index.
    """
    if not status:
        return ""
    env = _build_env()
    tool_list = ", ".join(f"`{n}`" for n in tool_names)
    return env.render_template("index_status", status=status, tool_list=tool_list)


def render_compact(extra_instructions: str = "") -> str:
    """Render the compaction system instructions."""
    env = _build_env()
    return env.render_template(
        "compact",
        extra_instructions=extra_instructions,
    )
