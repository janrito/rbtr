"""Tests for rbtr.prompts — system prompt rendering.

Uses shared review target constants so the test data is
inspectable and consistent across prompt tests.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rbtr.models import BranchTarget, PRTarget
from rbtr.prompts import render_review, render_system, review_tag

# ── Shared test data ─────────────────────────────────────────────────

_PR_FIX_BUG = PRTarget(
    number=42,
    title="Fix bug",
    author="alice",
    base_branch="main",
    head_branch="fix-bug",
    updated_at=datetime(2025, 1, 1, tzinfo=UTC),
)

_PR_WITH_BODY = PRTarget(
    number=99,
    title="Add feature",
    author="bob",
    body="This PR adds the frobnicator.\n\n## Changes\n- New module",
    base_branch="main",
    head_branch="add-feature",
    updated_at=datetime(2025, 1, 1, tzinfo=UTC),
)

_PR_EMPTY_BODY = PRTarget(
    number=99,
    title="Quick fix",
    author="bob",
    body="",
    base_branch="main",
    head_branch="quick-fix",
    updated_at=datetime(2025, 1, 1, tzinfo=UTC),
)

_BRANCH_TARGET = BranchTarget(
    base_branch="main",
    head_branch="feature-x",
    updated_at=datetime(2025, 1, 1, tzinfo=UTC),
)


def _make_session(**kwargs):  # type: ignore[no-untyped-def]
    from rbtr.engine import Session

    defaults = {
        "owner": "acme",
        "repo_name": "widgets",
    }
    defaults.update(kwargs)
    return Session(**defaults)


# ── System prompt ────────────────────────────────────────────────────


def test_render_system_no_target() -> None:
    session = _make_session()
    text = render_system(session)
    assert "acme/widgets" in text
    assert "(none selected)" in text


def test_render_system_pr_target() -> None:
    session = _make_session(review_target=_PR_FIX_BUG)
    text = render_system(session)
    assert "PR #42" in text
    assert "Fix bug" in text
    assert "alice" in text
    assert "`fix-bug`" in text


def test_render_system_pr_body() -> None:
    session = _make_session(review_target=_PR_WITH_BODY)
    text = render_system(session)
    assert "frobnicator" in text
    assert "## Changes" in text


def test_render_system_pr_empty_body() -> None:
    session = _make_session(review_target=_PR_EMPTY_BODY)
    text = render_system(session)
    assert "Quick fix" in text
    assert "Description" not in text


def test_render_system_branch_target() -> None:
    session = _make_session(review_target=_BRANCH_TARGET)
    text = render_system(session)
    assert "branch `feature-x`" in text


def test_render_system_contains_date() -> None:
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
    session = _make_session()
    text = render_review(session)
    for section in ("Design", "Correctness", "Readability", "Testing", "Security"):
        assert f"### {section}" in text


# ── Review tag ───────────────────────────────────────────────────────


def test_review_tag_pr() -> None:
    session = _make_session(review_target=_PR_FIX_BUG)
    assert review_tag(session) == "PR-42"


def test_review_tag_branch() -> None:
    session = _make_session(review_target=_BRANCH_TARGET)
    assert review_tag(session) == "feature-x"


def test_review_tag_no_target() -> None:
    session = _make_session()
    assert review_tag(session) == ""


def test_render_review_includes_tag_for_pr() -> None:
    session = _make_session(review_target=_PR_FIX_BUG)
    text = render_review(session)
    assert "PR-42" in text


def test_render_review_includes_tag_for_branch() -> None:
    session = _make_session(review_target=_BRANCH_TARGET)
    text = render_review(session)
    assert "feature-x" in text
