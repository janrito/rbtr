"""Tests for `read_file` with absolute skill paths."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.llm.tools.file import read_file
from rbtr.models import BranchTarget
from rbtr.skills.registry import Skill, SkillRegistry, SkillSource
from rbtr.state import EngineState
from tests.llm.conftest import FakeCtx


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """A skill directory with a SKILL.md and a helper script."""
    d = tmp_path / "my-skill"
    d.mkdir()
    (d / "SKILL.md").write_text("# My Skill\n")
    (d / "helper.sh").write_text("#!/bin/bash\necho hello\n")
    return d


@pytest.fixture
def state_with_skills(tmp_path: Path, skill_dir: Path) -> EngineState:
    """EngineState with a real git repo and a skill registry."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = pygit2.init_repository(str(repo_dir))
    sig = pygit2.Signature("test", "test@test.com")
    blob = repo.create_blob(b"content")
    tb = repo.TreeBuilder()
    tb.insert("dummy.txt", blob, pygit2.GIT_FILEMODE_BLOB)
    c = repo.create_commit("refs/heads/main", sig, sig, "init", tb.write(), [])
    repo.set_head("refs/heads/main")

    state = EngineState(repo=repo, owner="o", repo_name="r")
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="main",
        base_commit=str(c),
        head_commit=str(c),
        updated_at=0,
    )
    registry = SkillRegistry()
    registry.add(
        Skill(
            name="my-skill",
            description="Test.",
            file_path=str(skill_dir / "SKILL.md"),
            base_dir=str(skill_dir),
            source=SkillSource.USER,
        )
    )
    state.skill_registry = registry
    return state


def test_read_absolute_skill_file(state_with_skills: EngineState, skill_dir: Path) -> None:
    ctx = FakeCtx(state_with_skills)
    result = read_file(ctx, str(skill_dir / "SKILL.md"))  # type: ignore[arg-type]
    assert "My Skill" in result


def test_read_absolute_skill_helper(state_with_skills: EngineState, skill_dir: Path) -> None:
    ctx = FakeCtx(state_with_skills)
    result = read_file(ctx, str(skill_dir / "helper.sh"))  # type: ignore[arg-type]
    assert "echo hello" in result


def test_read_absolute_outside_skill_dir(state_with_skills: EngineState, tmp_path: Path) -> None:
    ctx = FakeCtx(state_with_skills)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    result = read_file(ctx, str(outside))  # type: ignore[arg-type]
    assert "not within a skill directory" in result


def test_read_absolute_no_registry(tmp_path: Path) -> None:
    """Absolute paths are rejected when no skill registry exists."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = pygit2.init_repository(str(repo_dir))
    sig = pygit2.Signature("test", "test@test.com")
    blob = repo.create_blob(b"x")
    tb = repo.TreeBuilder()
    tb.insert("f.txt", blob, pygit2.GIT_FILEMODE_BLOB)
    c = repo.create_commit("refs/heads/main", sig, sig, "init", tb.write(), [])
    state = EngineState(repo=repo, owner="o", repo_name="r")
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="main",
        base_commit=str(c),
        head_commit=str(c),
        updated_at=0,
    )
    ctx = FakeCtx(state)
    result = read_file(ctx, "/some/absolute/path.md")  # type: ignore[arg-type]
    assert "not within a skill directory" in result
