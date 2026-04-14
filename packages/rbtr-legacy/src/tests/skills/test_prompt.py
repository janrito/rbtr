"""Tests for skill catalog prompt rendering."""

from __future__ import annotations

from rbtr.prompts import render_skills
from rbtr.skills.registry import Skill, SkillSource


def _skill(name: str, description: str, file_path: str = "/s/SKILL.md") -> Skill:
    return Skill(
        name=name,
        description=description,
        file_path=file_path,
        base_dir="/s",
        source=SkillSource.USER,
    )


def test_contains_xml_structure() -> None:
    result = render_skills([_skill("brave-search", "Web search.", "/pi/brave-search/SKILL.md")])
    assert "<available_skills>" in result
    assert "</available_skills>" in result
    assert "<name>brave-search</name>" in result
    assert "<description>Web search.</description>" in result
    assert "<location>/pi/brave-search/SKILL.md</location>" in result


def test_preserves_special_chars() -> None:
    result = render_skills([_skill("test", 'Uses <brackets> & "quotes"')])
    assert '<brackets> & "quotes"' in result


def test_includes_preamble() -> None:
    result = render_skills([_skill("test", "A skill.")])
    assert "read_file" in result
    assert "skill directory" in result


def test_multiple_skills() -> None:
    skills = [_skill("alpha", "First.", "/a/SKILL.md"), _skill("beta", "Second.", "/b/SKILL.md")]
    result = render_skills(skills)
    assert result.count("<skill>") == 2
    assert "<name>alpha</name>" in result
    assert "<name>beta</name>" in result
