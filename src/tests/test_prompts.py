"""Tests for rbtr.prompts — persona and review instruction rendering.

Uses shared review target constants so the test data is
inspectable and consistent across prompt tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rbtr.models import BranchTarget, PRTarget
from rbtr.prompts import render_review, render_system
from rbtr.state import EngineState

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


def _make_engine_state(**kwargs: Any) -> EngineState:

    defaults = {
        "owner": "acme",
        "repo_name": "widgets",
    }
    defaults.update(kwargs)
    return EngineState(**defaults)


# ── Persona ──────────────────────────────────────────────────────────


def test_persona_contains_identity() -> None:
    text = render_system()
    assert "rbtr" in text
    assert "arbiter" in text


def test_persona_contains_authority() -> None:
    text = render_system()
    assert "reviewer has authority" in text


# ── Review — context rendering ───────────────────────────────────────


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


def test_review_pr_body() -> None:
    state = _make_engine_state(review_target=_PR_WITH_BODY)
    text = render_review(state)
    assert "frobnicator" in text
    assert "## Changes" in text


def test_review_pr_empty_body() -> None:
    state = _make_engine_state(review_target=_PR_EMPTY_BODY)
    text = render_review(state)
    assert "Quick fix" in text
    assert "Description" not in text


def test_review_branch_target() -> None:
    state = _make_engine_state(review_target=_BRANCH_TARGET)
    text = render_review(state)
    assert "branch `feature-x`" in text


def test_review_contains_date() -> None:
    state = _make_engine_state()
    text = render_review(state)
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    assert today in text


def test_review_unknown_repo() -> None:
    state = _make_engine_state(owner="", repo_name="")
    text = render_review(state)
    assert "unknown/unknown" in text


# ── Review — strategy sections ───────────────────────────────────────


def test_review_contains_key_sections() -> None:
    """Core sections are present in the review instructions."""
    state = _make_engine_state()
    text = render_review(state)
    for section in ("Principles", "Severity", "Two voices", "Format"):
        assert f"## {section}" in text


# ── SYSTEM.md override (persona only) ────────────────────────────────


def test_system_override_replaces_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When SYSTEM.md exists in RBTR_DIR, it replaces the built-in persona."""
    custom = "Custom persona."
    (tmp_path / "SYSTEM.md").write_text(custom)
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path)
    text = render_system()
    assert text == "Custom persona."


def test_system_override_absent_uses_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When SYSTEM.md does not exist, the built-in persona is used."""
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path)
    text = render_system()
    assert "rbtr" in text
    assert "arbiter" in text


def test_system_override_has_template_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom SYSTEM.md has access to persona template variables."""
    template = "{% if project_instructions %}PI: {{ project_instructions }}{% endif %}"
    (tmp_path / "SYSTEM.md").write_text(template)
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path)
    # No project instructions — should render empty.
    monkeypatch.chdir(tmp_path)
    text = render_system()
    assert text == ""


def test_system_override_does_not_affect_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SYSTEM.md override replaces persona only — review is unaffected."""
    (tmp_path / "SYSTEM.md").write_text("Custom persona only.")
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path)
    state = _make_engine_state(review_target=_PR_FIX_BUG)
    review = render_review(state)
    assert "PR #42" in review
    assert "Fix bug" in review


# ── APPEND_SYSTEM.md injection (persona) ─────────────────────────────


def test_append_system_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """APPEND_SYSTEM.md content appears in the persona."""
    (tmp_path / "APPEND_SYSTEM.md").write_text("Always check for nil pointers.")
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path)
    text = render_system()
    assert "Always check for nil pointers." in text


def test_append_system_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When APPEND_SYSTEM.md does not exist, no additional instructions section appears."""
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path)
    text = render_system()
    assert "Additional instructions" not in text


def test_append_system_with_custom_system(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """APPEND_SYSTEM.md is appended even when using a custom SYSTEM.md."""
    (tmp_path / "SYSTEM.md").write_text(
        "Custom.{% if append_system %}\n{{ append_system }}{% endif %}"
    )
    (tmp_path / "APPEND_SYSTEM.md").write_text("Extra rules.")
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path)
    text = render_system()
    assert "Custom." in text
    assert "Extra rules." in text


# ── Project instructions injection (persona) ─────────────────────────


def test_project_instructions_single_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single AGENTS.md from the repo root is injected into persona."""
    (tmp_path / "AGENTS.md").write_text("Use Go idioms.")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path / "config")
    text = render_system()
    assert "Use Go idioms." in text


def test_project_instructions_multiple_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple instruction files are concatenated in list order."""
    (tmp_path / "AGENTS.md").write_text("Rule one.")
    (tmp_path / "REVIEW.md").write_text("Rule two.")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path / "config")
    monkeypatch.setattr("rbtr.prompts.config.project_instructions", ["AGENTS.md", "REVIEW.md"])
    text = render_system()
    assert "Rule one." in text
    assert "Rule two." in text
    # Verify order: rule one before rule two.
    assert text.index("Rule one.") < text.index("Rule two.")


def test_project_instructions_missing_files_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing project instruction files are silently skipped."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path / "config")
    text = render_system()
    assert "Project instructions" not in text


def test_project_instructions_custom_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom filename list via config is respected."""
    (tmp_path / "CUSTOM.md").write_text("Custom rules.")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path / "config")
    monkeypatch.setattr("rbtr.prompts.config.project_instructions", ["CUSTOM.md"])
    text = render_system()
    assert "Custom rules." in text


def test_project_instructions_empty_file_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty or whitespace-only instruction file is skipped."""
    (tmp_path / "AGENTS.md").write_text("   \n  ")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path / "config")
    text = render_system()
    assert "Project instructions" not in text


# ── Combined scenarios ───────────────────────────────────────────────


def test_all_three_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SYSTEM.md override + project instructions + APPEND_SYSTEM.md."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    (config_dir / "SYSTEM.md").write_text(
        "Base."
        "{% if project_instructions %}\nProject: {{ project_instructions }}{% endif %}"
        "{% if append_system %}\nAppend: {{ append_system }}{% endif %}"
    )
    (config_dir / "APPEND_SYSTEM.md").write_text("User extra.")
    (repo_dir / "AGENTS.md").write_text("Project rules.")

    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", config_dir)
    monkeypatch.chdir(repo_dir)
    text = render_system()
    assert "Base." in text
    assert "Project rules." in text
    assert "User extra." in text


def test_none_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No overrides — built-in persona, no project instructions, no append."""
    monkeypatch.setattr("rbtr.prompts.RBTR_DIR", tmp_path / "empty")
    monkeypatch.chdir(tmp_path)
    text = render_system()
    assert "rbtr" in text
    assert "arbiter" in text
    assert "Project instructions" not in text
    assert "Additional instructions" not in text
