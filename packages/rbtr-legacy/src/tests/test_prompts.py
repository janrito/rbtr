"""Tests for rbtr.prompts — template loading and variable injection mechanics.

Does NOT test prompt wording — only that templates load, variables
inject, overrides replace, and conditionals branch correctly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rbtr.config import config
from rbtr.models import BranchTarget, PRTarget, Target
from rbtr.prompts import render_review, render_system
from rbtr.state import EngineState

# ── Shared test data ─────────────────────────────────────────────────

_PR_FIX_BUG = PRTarget(
    number=42,
    title="Fix bug",
    author="alice",
    base_branch="main",
    head_branch="fix-bug",
    base_commit="main",
    head_commit="fix-bug",
    updated_at=datetime(2025, 1, 1, tzinfo=UTC),
)

_PR_WITH_BODY = PRTarget(
    number=99,
    title="Add feature",
    author="bob",
    body="This PR adds the frobnicator.\n\n## Changes\n- New module",
    base_branch="main",
    head_branch="add-feature",
    base_commit="main",
    head_commit="add-feature",
    updated_at=datetime(2025, 1, 1, tzinfo=UTC),
)

_PR_EMPTY_BODY = PRTarget(
    number=99,
    title="Quick fix",
    author="bob",
    body="",
    base_branch="main",
    head_branch="quick-fix",
    base_commit="main",
    head_commit="quick-fix",
    updated_at=datetime(2025, 1, 1, tzinfo=UTC),
)

_BRANCH_TARGET = BranchTarget(
    base_branch="main",
    head_branch="feature-x",
    base_commit="main",
    head_commit="feature-x",
    updated_at=datetime(2025, 1, 1, tzinfo=UTC),
)


def _make_engine_state(
    *,
    owner: str = "acme",
    repo_name: str = "widgets",
    review_target: Target | None = None,
) -> EngineState:
    state = EngineState(owner=owner, repo_name=repo_name)
    state.review_target = review_target
    return state


# ── System — builtin loads ───────────────────────────────────────────


def test_system_renders() -> None:
    """Built-in system template loads and renders without error."""
    text = render_system()
    assert text


# ── Review — context variable injection ──────────────────────────────


def test_review_no_target() -> None:
    state = _make_engine_state()
    text = render_review(state)
    assert "acme/widgets" in text
    assert "(none selected)" in text


def test_review_pr_target() -> None:
    state = _make_engine_state(review_target=_PR_FIX_BUG)
    text = render_review(state)
    assert "PR #42" in text
    assert "Fix bug" in text
    assert "alice" in text
    assert "`fix-bug`" in text


def test_review_pr_body_rendered() -> None:
    state = _make_engine_state(review_target=_PR_WITH_BODY)
    text = render_review(state)
    assert "frobnicator" in text


def test_review_pr_empty_body_omits_description() -> None:
    state = _make_engine_state(review_target=_PR_EMPTY_BODY)
    text = render_review(state)
    assert "Description" not in text


def test_review_branch_target() -> None:
    state = _make_engine_state(review_target=_BRANCH_TARGET)
    text = render_review(state)
    assert "branch `feature-x`" in text


def test_review_injects_date() -> None:
    state = _make_engine_state()
    text = render_review(state)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    assert today in text


def test_review_unknown_repo_fallback() -> None:
    state = _make_engine_state(owner="", repo_name="")
    text = render_review(state)
    assert "unknown/unknown" in text


# ── SYSTEM.md override ───────────────────────────────────────────────


def test_system_override_replaces_builtin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (Path(config.user_dir) / "SYSTEM.md").write_text("Custom persona.")
    assert render_system() == "Custom persona."


def test_system_override_receives_template_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (Path(config.user_dir) / "SYSTEM.md").write_text(
        "{% if project_instructions %}PI: {{ project_instructions }}{% endif %}"
    )
    monkeypatch.chdir(tmp_path)
    # No project instructions — conditional should produce empty.
    assert render_system() == ""


def test_system_override_does_not_affect_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (Path(config.user_dir) / "SYSTEM.md").write_text("Custom persona only.")
    state = _make_engine_state(review_target=_PR_FIX_BUG)
    review = render_review(state)
    assert "PR #42" in review


# ── APPEND_SYSTEM.md injection ───────────────────────────────────────


def test_append_system_injected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (Path(config.user_dir) / "APPEND_SYSTEM.md").write_text("Always check for nil pointers.")
    assert "Always check for nil pointers." in render_system()


def test_append_system_absent_no_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "Additional instructions" not in render_system()


def test_append_system_works_with_custom_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (Path(config.user_dir) / "SYSTEM.md").write_text(
        "Custom.{% if append_system %}\n{{ append_system }}{% endif %}"
    )
    (Path(config.user_dir) / "APPEND_SYSTEM.md").write_text("Extra rules.")
    text = render_system()
    assert "Custom." in text
    assert "Extra rules." in text


# ── Project instructions injection ───────────────────────────────────


def test_project_instructions_single_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "AGENTS.md").write_text("Use Go idioms.")
    monkeypatch.chdir(tmp_path)
    assert "Use Go idioms." in render_system()


def test_project_instructions_multiple_concatenated_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("Rule one.")
    (tmp_path / "REVIEW.md").write_text("Rule two.")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rbtr.prompts.config.project_instructions", ["AGENTS.md", "REVIEW.md"])
    text = render_system()
    assert text.index("Rule one.") < text.index("Rule two.")


def test_project_instructions_missing_files_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert "Project instructions" not in render_system()


def test_project_instructions_custom_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "CUSTOM.md").write_text("Custom rules.")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rbtr.prompts.config.project_instructions", ["CUSTOM.md"])
    assert "Custom rules." in render_system()


def test_project_instructions_empty_file_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("   \n  ")
    monkeypatch.chdir(tmp_path)
    assert "Project instructions" not in render_system()


# ── Combined scenarios ───────────────────────────────────────────────


def test_all_injection_sources_combined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    (Path(config.user_dir) / "SYSTEM.md").write_text(
        "Base."
        "{% if project_instructions %}\nProject: {{ project_instructions }}{% endif %}"
        "{% if append_system %}\nAppend: {{ append_system }}{% endif %}"
    )
    (Path(config.user_dir) / "APPEND_SYSTEM.md").write_text("User extra.")
    (repo_dir / "AGENTS.md").write_text("Project rules.")

    monkeypatch.chdir(repo_dir)
    text = render_system()
    assert "Base." in text
    assert "Project rules." in text
    assert "User extra." in text


def test_no_overrides_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    text = render_system()
    assert text
    assert "Project instructions" not in text
    assert "Additional instructions" not in text
