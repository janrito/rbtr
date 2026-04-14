"""Tests for the skill registry."""

from __future__ import annotations

from rbtr_legacy.skills.registry import Skill, SkillRegistry, SkillSource


def _skill(
    name: str = "test-skill",
    description: str = "A test skill.",
    file_path: str = "/skills/test-skill/SKILL.md",
    base_dir: str = "/skills/test-skill",
    source: SkillSource = SkillSource.PROJECT,
    disable_model_invocation: bool = False,
) -> Skill:
    return Skill(
        name=name,
        description=description,
        file_path=file_path,
        base_dir=base_dir,
        source=source,
        disable_model_invocation=disable_model_invocation,
    )


def test_add_and_get() -> None:
    reg = SkillRegistry()
    skill = _skill()
    reg.add(skill)
    assert reg.get("test-skill") is skill
    assert len(reg) == 1


def test_get_missing_returns_none() -> None:
    reg = SkillRegistry()
    assert reg.get("nope") is None


def test_first_wins_on_collision() -> None:
    reg = SkillRegistry()
    first = _skill(file_path="/a/SKILL.md", base_dir="/a")
    second = _skill(file_path="/b/SKILL.md", base_dir="/b")
    reg.add(first)
    reg.add(second)
    assert reg.get("test-skill") is first
    assert len(reg) == 1


def test_visible_excludes_hidden() -> None:
    reg = SkillRegistry()
    reg.add(_skill(name="visible", file_path="/v/SKILL.md", base_dir="/v"))
    reg.add(
        _skill(
            name="hidden",
            file_path="/h/SKILL.md",
            base_dir="/h",
            disable_model_invocation=True,
        )
    )
    visible = reg.visible()
    assert len(visible) == 1
    assert visible[0].name == "visible"


def test_all_includes_hidden() -> None:
    reg = SkillRegistry()
    reg.add(_skill(name="visible", file_path="/v/SKILL.md", base_dir="/v"))
    reg.add(
        _skill(
            name="hidden",
            file_path="/h/SKILL.md",
            base_dir="/h",
            disable_model_invocation=True,
        )
    )
    assert len(reg.all()) == 2


def test_empty_registry_is_falsy() -> None:
    reg = SkillRegistry()
    assert not reg
    assert len(reg) == 0


def test_populated_registry_is_truthy() -> None:
    reg = SkillRegistry()
    reg.add(_skill())
    assert reg


# ── match_command ────────────────────────────────────────────────────


def test_match_command_hit() -> None:
    reg = SkillRegistry()
    reg.add(_skill(name="brave", base_dir="/home/u/.pi/skills/brave"))
    result = reg.match_command('/home/u/.pi/skills/brave/search.sh "query"')
    assert result is not None
    skill, rel = result
    assert skill.name == "brave"
    assert rel == 'search.sh "query"'


def test_match_command_miss() -> None:
    reg = SkillRegistry()
    reg.add(_skill(name="brave", base_dir="/home/u/.pi/skills/brave"))
    assert reg.match_command("echo hello") is None


def test_match_command_empty_registry() -> None:
    reg = SkillRegistry()
    assert reg.match_command("/any/path/script.sh") is None


def test_match_command_longest_prefix_wins() -> None:
    """When a skill is nested inside another's base_dir, the deeper one wins."""
    reg = SkillRegistry()
    reg.add(_skill(name="parent", base_dir="/skills/tools", file_path="/skills/tools/SKILL.md"))
    reg.add(
        _skill(
            name="child",
            base_dir="/skills/tools/sub",
            file_path="/skills/tools/sub/SKILL.md",
        )
    )
    result = reg.match_command("/skills/tools/sub/run.sh arg")
    assert result is not None
    assert result[0].name == "child"
    assert result[1] == "run.sh arg"


def test_match_command_no_partial_dir_match() -> None:
    """base_dir '/skills/brave' must not match '/skills/bravery/x.sh'."""
    reg = SkillRegistry()
    reg.add(_skill(name="brave", base_dir="/skills/brave"))
    assert reg.match_command("/skills/bravery/x.sh") is None
