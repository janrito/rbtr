"""Prompt rendering — loads markdown templates and fills placeholders."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from importlib import resources
from typing import TYPE_CHECKING, Any

import minijinja

from rbtr.models import BranchTarget, PRTarget

if TYPE_CHECKING:
    from rbtr.engine import Session


def _load_template(name: str) -> str:
    """Read a .md template from the prompts package."""
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")  # type: ignore[union-attr]  # Traversable always has read_text when joinpath succeeds


def _build_env() -> minijinja.Environment:
    """Create a MiniJinja environment with all prompt templates."""
    env = minijinja.Environment()
    env.add_template("system", _load_template("system.md"))
    env.add_template("review", _load_template("review.md"))
    env.add_template("index_status", _load_template("index_status.md"))
    env.add_template("compact", _load_template("compact.md"))
    return env


def _context(session: Session) -> dict[str, Any]:
    """Build template context from live session state."""
    ctx: dict[str, Any] = {
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "owner": session.owner or "unknown",
        "repo": session.repo_name or "unknown",
        "target_kind": "none",
        "base_branch": "",
        "branch": "",
        "pr_number": 0,
        "pr_title": "",
        "pr_author": "",
        "pr_body": "",
    }

    match session.review_target:
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


def review_tag(session: Session) -> str:
    """Derive a short tag from the review target for file naming.

    Examples: ``PR-42`` for a pull request, ``fix-auth`` for a branch.
    Returns ``""`` when no target is selected.
    """
    match session.review_target:
        case PRTarget(number=n):
            return f"PR-{n}"
        case BranchTarget(head_branch=head):
            # Sanitise: keep only alphanumeric, hyphens, dots.
            tag = re.sub(r"[^a-zA-Z0-9._-]", "-", head)
            # Collapse runs of hyphens and strip leading/trailing.
            tag = re.sub(r"-{2,}", "-", tag).strip("-")
            return tag or "branch"
    return ""


def render_system(session: Session) -> str:
    """Render the system prompt with live session data."""
    env = _build_env()
    return env.render_template("system", **_context(session))


def render_review(session: Session) -> str:
    """Render the review guidelines with session context."""
    env = _build_env()
    return env.render_template(
        "review",
        review_tag=review_tag(session),
    )


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
