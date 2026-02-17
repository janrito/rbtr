"""Tests for rbtr.prompts — system prompt rendering."""

from __future__ import annotations

from rbtr.models import BranchTarget, PRTarget
from rbtr.prompts import render_review, render_system


def _make_session(**kwargs):  # type: ignore[no-untyped-def]
    from rbtr.engine import Session

    defaults = {
        "owner": "acme",
        "repo_name": "widgets",
    }
    defaults.update(kwargs)
    return Session(**defaults)


def test_render_system_no_target() -> None:
    session = _make_session()
    text = render_system(session)
    assert "acme/widgets" in text
    assert "(none selected)" in text


def test_render_system_pr_target() -> None:
    from datetime import UTC, datetime

    target = PRTarget(
        number=42,
        title="Fix bug",
        author="alice",
        head_branch="fix-bug",
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    session = _make_session(review_target=target)
    text = render_system(session)
    assert "PR #42" in text
    assert "Fix bug" in text
    assert "alice" in text
    assert "`fix-bug`" in text


def test_render_system_branch_target() -> None:
    from datetime import UTC, datetime

    target = BranchTarget(
        head_branch="feature-x",
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    session = _make_session(review_target=target)
    text = render_system(session)
    assert "branch `feature-x`" in text


def test_render_system_contains_date() -> None:
    from datetime import UTC, datetime

    session = _make_session()
    text = render_system(session)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    assert today in text


def test_render_system_unknown_repo() -> None:
    session = _make_session(owner="", repo_name="")
    text = render_system(session)
    assert "unknown/unknown" in text


# ── Review guidelines ────────────────────────────────────────────────


def test_render_review_contains_sections() -> None:
    text = render_review()
    for section in ("Design", "Correctness", "Clarity", "Maintenance"):
        assert f"## {section}" in text
