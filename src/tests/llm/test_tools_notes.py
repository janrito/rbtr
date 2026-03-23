"""Tests for notes tool — edit."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import RunContext

from rbtr.llm.deps import AgentDeps
from rbtr.llm.tools.notes import edit
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

from .ctx import build_tool_ctx

# ── edit tool ────────────────────────────────────────────────────────


@pytest.fixture
def ctx(store: SessionStore) -> RunContext[AgentDeps]:
    """Empty-state context — edit doesn't need repo or index."""
    return build_tool_ctx(EngineState(), store)


def test_edit_create_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: RunContext[AgentDeps]
) -> None:
    """Creating a new notes file writes content and returns confirmation."""
    monkeypatch.chdir(tmp_path)
    result = edit(ctx, ".rbtr/notes/plan.md", "# Plan\n\n- Step 1\n")
    assert "Created" in result
    content = (tmp_path / ".rbtr" / "notes" / "plan.md").read_text()
    assert content == "# Plan\n\n- Step 1\n"


def test_edit_append_to_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: RunContext[AgentDeps]
) -> None:
    """Appending to an existing file concatenates content."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr" / "notes").mkdir(parents=True)
    (tmp_path / ".rbtr" / "notes" / "notes.md").write_text("# Notes\n")
    result = edit(ctx, ".rbtr/notes/notes.md", "\n- Finding 1\n")
    assert "Appended" in result
    content = (tmp_path / ".rbtr" / "notes" / "notes.md").read_text()
    assert content == "# Notes\n\n- Finding 1\n"


def test_edit_replace_exact_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: RunContext[AgentDeps]
) -> None:
    """Exact old_text match is replaced with new_text."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr" / "notes").mkdir(parents=True)
    original = "# Plan\n\n- [ ] Review handler\n- [ ] Review config\n"
    (tmp_path / ".rbtr" / "notes" / "plan.md").write_text(original)
    result = edit(
        ctx,
        ".rbtr/notes/plan.md",
        "- [x] Review handler  ✓\n",
        old_text="- [ ] Review handler\n",
    )
    assert "Replaced" in result
    content = (tmp_path / ".rbtr" / "notes" / "plan.md").read_text()
    assert "- [x] Review handler  ✓" in content
    assert "- [ ] Review config" in content  # untouched


def test_edit_replace_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: RunContext[AgentDeps]
) -> None:
    """old_text that doesn't exist returns an error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr" / "notes").mkdir(parents=True)
    (tmp_path / ".rbtr" / "notes" / "plan.md").write_text("# Plan\n")
    result = edit(
        ctx,
        ".rbtr/notes/plan.md",
        "replacement",
        old_text="nonexistent text",
    )
    assert "not found" in result


def test_edit_replace_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: RunContext[AgentDeps]
) -> None:
    """old_text matching multiple times returns an error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr" / "notes").mkdir(parents=True)
    (tmp_path / ".rbtr" / "notes" / "plan.md").write_text("AAA\nBBB\nAAA\n")
    result = edit(
        ctx,
        ".rbtr/notes/plan.md",
        "CCC",
        old_text="AAA",
    )
    assert "2 times" in result


def test_edit_replace_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: RunContext[AgentDeps]
) -> None:
    """Replacing in a nonexistent file returns an error."""
    monkeypatch.chdir(tmp_path)
    result = edit(
        ctx,
        ".rbtr/notes/plan.md",
        "new",
        old_text="old",
    )
    assert "does not exist" in result


def test_edit_rejects_path_outside_editable(ctx: RunContext[AgentDeps]) -> None:
    """Paths not matching any editable_include pattern are rejected."""
    result = edit(ctx, "src/main.py", "content")
    assert "editable_include" in result


def test_edit_rejects_drafts_dir(ctx: RunContext[AgentDeps]) -> None:
    """Paths inside .rbtr/drafts/ are rejected — use draft tools."""
    result = edit(ctx, ".rbtr/drafts/42.yaml", "content")
    assert "editable_include" in result


def test_edit_rejects_path_traversal(ctx: RunContext[AgentDeps]) -> None:
    """Paths with '..' are rejected."""
    result = edit(ctx, ".rbtr/notes/../escape.md", "content")
    assert "not allowed" in result


def test_edit_allows_editable_include(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: RunContext[AgentDeps]
) -> None:
    """Paths matching `editable_include` patterns are writable."""
    monkeypatch.chdir(tmp_path)
    result = edit(ctx, ".rbtr/AGENTS.md", "# Agent rules\n")
    assert "Created" in result
    content = (tmp_path / ".rbtr" / "AGENTS.md").read_text()
    assert content == "# Agent rules\n"


def test_edit_replace_editable_include(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: RunContext[AgentDeps]
) -> None:
    """Replacing text in an editable_include file works."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr").mkdir(parents=True)
    (tmp_path / ".rbtr" / "AGENTS.md").write_text("# Old title\n")
    result = edit(
        ctx,
        ".rbtr/AGENTS.md",
        "# New title\n",
        old_text="# Old title\n",
    )
    assert "Replaced" in result
    content = (tmp_path / ".rbtr" / "AGENTS.md").read_text()
    assert "# New title" in content


def test_edit_rejects_editable_include_mismatch(ctx: RunContext[AgentDeps]) -> None:
    """Paths that don't match any editable_include pattern are rejected."""
    result = edit(ctx, ".rbtr/config.toml", "content")
    assert "editable_include" in result


def test_edit_creates_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: RunContext[AgentDeps]
) -> None:
    """Nested paths under .rbtr/notes/ create intermediate directories."""
    monkeypatch.chdir(tmp_path)
    result = edit(ctx, ".rbtr/notes/sub/comments.md", "# Draft\n")
    assert "Created" in result
    assert (tmp_path / ".rbtr" / "notes" / "sub" / "comments.md").exists()
