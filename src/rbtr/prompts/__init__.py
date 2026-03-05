"""Prompt rendering — loads markdown templates and fills placeholders.

Three templates, two concerns:

- **System** (``system.md``) — identity, language, project
  instructions.  Shared by the main agent and the compaction
  agent.  No runtime state needed.
- **Review task** (``review.md``) — review context, principles,
  strategy, two voices, format.  Main agent only.
- **Compaction task** (``compact.md``) — what to preserve and
  drop when summarising history.  Compaction agent only.

Project-level instructions (``AGENTS.md`` etc.) and user-level
extensions (``APPEND_SYSTEM.md``) are injected as plain text
via template variables — they are not themselves templates.
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
    """Read the append-system file from ``~/.config/rbtr/`` if configured."""
    name = config.append_system
    if not name:
        return ""
    return _read_optional(RBTR_DIR / name)


def _load_system_override() -> str:
    """Read the system prompt override from ``~/.config/rbtr/`` if configured.

    When present, replaces the built-in system template.
    Review and compaction instructions are unaffected.
    """
    name = config.system_prompt_override
    if not name:
        return ""
    return _read_optional(RBTR_DIR / name)


def _render(template: str, **ctx: Any) -> str:
    """Render a minijinja template string with the given context."""
    env = minijinja.Environment()
    env.add_template("t", template)
    return env.render_template("t", **ctx)


# ── System ───────────────────────────────────────────────────────────


def render_system() -> str:
    """Render the system prompt (identity, language, project rules).

    If ``~/.config/rbtr/SYSTEM.md`` exists, it replaces the
    built-in template.  The same template variables are available.
    """
    template = _load_system_override() or _load_template("system.md")
    return _render(
        template,
        project_instructions=_load_project_instructions(),
        append_system=_load_append_system(),
    )


# ── Review task ──────────────────────────────────────────────────────


def _review_context(state: EngineState) -> dict[str, Any]:
    """Build template context for the review instructions."""
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
        "editable_globs": config.tools.editable_include,
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


def render_review(state: EngineState) -> str:
    """Render the review task instructions with live state data."""
    return _render(_load_template("review.md"), **_review_context(state))


# ── Index status ─────────────────────────────────────────────────────


def render_index_status(*, status: str, tool_names: list[str]) -> str:
    """Render the index status instruction.

    Args:
        status: ``"ready"``, ``"building"``, or ``""`` (no review target).
        tool_names: Names of tools that require the index.
    """
    if not status:
        return ""
    tool_list = ", ".join(f"`{n}`" for n in tool_names)
    return _render(_load_template("index_status.md"), status=status, tool_list=tool_list)


# ── Compaction task ──────────────────────────────────────────────────


def render_compact() -> str:
    """Render the compaction task instructions."""
    return _load_template("compact.md").strip()
