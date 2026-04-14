"""Tests for skill discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr_legacy.config import SkillsConfig
from rbtr_legacy.skills.discovery import _parse_skill_file, _scan_dir, load_skills
from rbtr_legacy.skills.registry import SkillRegistry, SkillSource


@pytest.fixture
def skill_tree(tmp_path: Path) -> Path:
    """Build a realistic skill directory tree.

    Layout:
        skills/
            standalone.md           (root .md file)
            brave-search/
                SKILL.md            (valid skill)
            vscode/
                SKILL.md            (valid skill)
            hidden/
                SKILL.md            (disable-model-invocation)
            no-desc/
                SKILL.md            (missing description — skipped)
            nested/
                deep/
                    SKILL.md        (nested skill)
            .dotdir/
                SKILL.md            (should be skipped — dotdir)
    """
    root = tmp_path / "skills"
    root.mkdir()

    # Root-level .md file.
    (root / "standalone.md").write_text(
        "---\nname: standalone\ndescription: A standalone skill.\n---\n# Standalone\n"
    )

    # Standard subdirectory skills.
    _write_skill(root / "brave-search", "brave-search", "Web search via Brave API.")
    _write_skill(root / "vscode", "vscode", "VS Code integration.")

    # Hidden skill.
    hidden = root / "hidden"
    hidden.mkdir()
    (hidden / "SKILL.md").write_text(
        "---\nname: hidden\ndescription: Hidden skill.\n"
        "disable-model-invocation: true\n---\n# Hidden\n"
    )

    # No description — should be skipped.
    no_desc = root / "no-desc"
    no_desc.mkdir()
    (no_desc / "SKILL.md").write_text("---\nname: no-desc\n---\n# No desc\n")

    # Deeply nested.
    deep = root / "nested" / "deep"
    deep.mkdir(parents=True)
    _write_skill(deep, "deep-skill", "A deeply nested skill.", parent=deep)

    # Dotdir — should be skipped.
    dotdir = root / ".dotdir"
    dotdir.mkdir()
    _write_skill(dotdir, "dotdir-skill", "Should not appear.", parent=dotdir)

    return root


def _write_skill(
    directory: Path,
    name: str,
    description: str,
    *,
    parent: Path | None = None,
) -> None:
    target = parent or directory
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n"
    )


# ── _scan_dir ────────────────────────────────────────────────────────


def test_scan_discovers_root_md_and_subdirs(skill_tree: Path) -> None:
    reg = SkillRegistry()
    _scan_dir(skill_tree, SkillSource.USER, reg)
    names = {s.name for s in reg.all()}
    assert "standalone" in names
    assert "brave-search" in names
    assert "vscode" in names
    assert "hidden" in names


def test_scan_finds_nested_skills(skill_tree: Path) -> None:
    reg = SkillRegistry()
    _scan_dir(skill_tree, SkillSource.USER, reg)
    assert reg.get("deep-skill") is not None


def test_scan_skips_dotdirs(skill_tree: Path) -> None:
    reg = SkillRegistry()
    _scan_dir(skill_tree, SkillSource.USER, reg)
    names = {s.name for s in reg.all()}
    assert "dotdir-skill" not in names


def test_scan_skips_missing_description(skill_tree: Path) -> None:
    reg = SkillRegistry()
    _scan_dir(skill_tree, SkillSource.USER, reg)
    assert reg.get("no-desc") is None


def test_scan_nonexistent_dir_is_noop() -> None:
    reg = SkillRegistry()
    _scan_dir(Path("/nonexistent/path"), SkillSource.USER, reg)
    assert len(reg) == 0


# ── _parse_skill_file ─────────────────────────────────────────────────


def test_parse_uses_parent_dir_name_when_no_name(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\ndescription: Fallback name test.\n---\n# Content\n")
    skill = _parse_skill_file(skill_dir / "SKILL.md", SkillSource.PROJECT)
    assert skill is not None
    assert skill.name == "my-skill"


def test_parse_sets_base_dir(tmp_path: Path) -> None:
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: Base dir test.\n---\n")
    skill = _parse_skill_file(skill_dir / "SKILL.md", SkillSource.USER)
    assert skill is not None
    assert skill.base_dir == str(skill_dir)


def test_parse_disable_model_invocation(tmp_path: Path) -> None:
    skill_dir = tmp_path / "hidden"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: hidden\ndescription: Hidden.\ndisable-model-invocation: true\n---\n"
    )
    skill = _parse_skill_file(skill_dir / "SKILL.md", SkillSource.USER)
    assert skill is not None
    assert skill.disable_model_invocation is True


def test_parse_missing_description_returns_none(tmp_path: Path) -> None:
    skill_dir = tmp_path / "no-desc"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\n# No desc\n")
    assert _parse_skill_file(skill_dir / "SKILL.md", SkillSource.USER) is None


# ── load_skills ──────────────────────────────────────────────────────


def test_load_skills_extra_dirs(skill_tree: Path) -> None:
    cfg = SkillsConfig(project_dirs=[], user_dirs=[], extra_dirs=[str(skill_tree)])
    reg = load_skills(cfg)
    # Should find skills from the extra dir.
    assert reg.get("brave-search") is not None


def test_load_skills_empty_when_no_dirs() -> None:
    cfg = SkillsConfig(project_dirs=[], user_dirs=[], extra_dirs=["/nonexistent"])
    reg = load_skills(cfg)
    assert len(reg) == 0
