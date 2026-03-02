"""Tests for notes tool — edit."""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.llm.tools.notes import edit
from rbtr.state import EngineState

from .conftest import FakeCtx

# ── edit tool ────────────────────────────────────────────────────────


def _edit_ctx() -> FakeCtx:
    """Minimal context — edit doesn't need repo or index."""
    state = EngineState()
    return FakeCtx(state)


def test_edit_create_new_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Creating a new notes file writes content and returns confirmation."""
    monkeypatch.chdir(tmp_path)
    ctx = _edit_ctx()
    result = edit(ctx, ".rbtr/notes/plan.md", "# Plan\n\n- Step 1\n")  # type: ignore[arg-type]
    assert "Created" in result
    content = (tmp_path / ".rbtr" / "notes" / "plan.md").read_text()
    assert content == "# Plan\n\n- Step 1\n"


def test_edit_append_to_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Appending to an existing file concatenates content."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr" / "notes").mkdir(parents=True)
    (tmp_path / ".rbtr" / "notes" / "notes.md").write_text("# Notes\n")
    ctx = _edit_ctx()
    result = edit(ctx, ".rbtr/notes/notes.md", "\n- Finding 1\n")  # type: ignore[arg-type]
    assert "Appended" in result
    content = (tmp_path / ".rbtr" / "notes" / "notes.md").read_text()
    assert content == "# Notes\n\n- Finding 1\n"


def test_edit_replace_exact_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact old_text match is replaced with new_text."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr" / "notes").mkdir(parents=True)
    original = "# Plan\n\n- [ ] Review handler\n- [ ] Review config\n"
    (tmp_path / ".rbtr" / "notes" / "plan.md").write_text(original)
    ctx = _edit_ctx()
    result = edit(
        ctx,  # type: ignore[arg-type]
        ".rbtr/notes/plan.md",
        "- [x] Review handler  ✓\n",
        old_text="- [ ] Review handler\n",
    )
    assert "Replaced" in result
    content = (tmp_path / ".rbtr" / "notes" / "plan.md").read_text()
    assert "- [x] Review handler  ✓" in content
    assert "- [ ] Review config" in content  # untouched


def test_edit_replace_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """old_text that doesn't exist returns an error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr" / "notes").mkdir(parents=True)
    (tmp_path / ".rbtr" / "notes" / "plan.md").write_text("# Plan\n")
    ctx = _edit_ctx()
    result = edit(
        ctx,  # type: ignore[arg-type]
        ".rbtr/notes/plan.md",
        "replacement",
        old_text="nonexistent text",
    )
    assert "not found" in result


def test_edit_replace_ambiguous(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """old_text matching multiple times returns an error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr" / "notes").mkdir(parents=True)
    (tmp_path / ".rbtr" / "notes" / "plan.md").write_text("AAA\nBBB\nAAA\n")
    ctx = _edit_ctx()
    result = edit(
        ctx,  # type: ignore[arg-type]
        ".rbtr/notes/plan.md",
        "CCC",
        old_text="AAA",
    )
    assert "2 times" in result


def test_edit_replace_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing in a nonexistent file returns an error."""
    monkeypatch.chdir(tmp_path)
    ctx = _edit_ctx()
    result = edit(
        ctx,  # type: ignore[arg-type]
        ".rbtr/notes/plan.md",
        "new",
        old_text="old",
    )
    assert "does not exist" in result


def test_edit_rejects_path_outside_notes() -> None:
    """Paths not inside .rbtr/notes/ are rejected."""
    ctx = _edit_ctx()
    result = edit(ctx, "src/main.py", "content")  # type: ignore[arg-type]
    assert "must be inside" in result


def test_edit_rejects_drafts_dir() -> None:
    """Paths inside .rbtr/drafts/ are rejected — use draft tools."""
    ctx = _edit_ctx()
    result = edit(ctx, ".rbtr/drafts/42.yaml", "content")  # type: ignore[arg-type]
    assert "must be inside" in result


def test_edit_rejects_path_traversal() -> None:
    """Paths with '..' are rejected."""
    ctx = _edit_ctx()
    result = edit(ctx, ".rbtr/notes/../escape.md", "content")  # type: ignore[arg-type]
    assert "not allowed" in result


def test_edit_creates_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested paths under .rbtr/notes/ create intermediate directories."""
    monkeypatch.chdir(tmp_path)
    ctx = _edit_ctx()
    result = edit(ctx, ".rbtr/notes/sub/comments.md", "# Draft\n")  # type: ignore[arg-type]
    assert "Created" in result
    assert (tmp_path / ".rbtr" / "notes" / "sub" / "comments.md").exists()
