"""Prompt rendering — loads markdown templates and fills placeholders."""

from __future__ import annotations

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
    return env


def _context(session: Session) -> dict[str, Any]:
    """Build template context from live session state."""
    ctx: dict[str, Any] = {
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "owner": session.owner or "unknown",
        "repo": session.repo_name or "unknown",
        "target_kind": "none",
        "branch": "",
        "pr_number": 0,
        "pr_title": "",
        "pr_author": "",
    }

    match session.review_target:
        case PRTarget(
            number=n,
            title=t,
            head_branch=b,
            author=a,
        ):
            ctx |= {
                "target_kind": "pr",
                "branch": b,
                "pr_number": n,
                "pr_title": t,
                "pr_author": a,
            }
        case BranchTarget(head_branch=b):
            ctx |= {
                "target_kind": "branch",
                "branch": b,
            }

    return ctx


def render_system(session: Session) -> str:
    """Render the system prompt with live session data."""
    env = _build_env()
    return env.render_template("system", **_context(session))


def render_review() -> str:
    """Render the review guidelines (static, no placeholders)."""
    env = _build_env()
    return env.render_template("review")
