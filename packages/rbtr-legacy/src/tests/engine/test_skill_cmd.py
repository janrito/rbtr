"""Tests for `/skill` — list and load skills."""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.engine.core import Engine
from rbtr.engine.types import TaskType
from rbtr.events import AutoSubmit, ContextMarkerReady
from rbtr.skills.registry import Skill, SkillRegistry, SkillSource
from tests.helpers import drain, has_event_type, output_texts


def _skill(
    name: str,
    description: str,
    *,
    file_path: str = "/fake/SKILL.md",
    hidden: bool = False,
) -> Skill:
    return Skill(
        name=name,
        description=description,
        file_path=file_path,
        base_dir="/fake",
        source=SkillSource.USER,
        disable_model_invocation=hidden,
    )


def _registry(*skills: Skill) -> SkillRegistry:
    reg = SkillRegistry()
    for s in skills:
        reg.add(s)
    return reg


# ── No skills ────────────────────────────────────────────────────────


def test_skill_no_registry(engine: Engine) -> None:
    """Warns when no skills are discovered."""
    engine.run_task(TaskType.COMMAND, "/skill")
    texts = output_texts(drain(engine.events))
    assert any("No skills" in t for t in texts)


def test_skill_empty_registry(engine: Engine) -> None:
    """Warns when registry exists but is empty."""
    engine.state.skill_registry = SkillRegistry()
    engine.run_task(TaskType.COMMAND, "/skill")
    texts = output_texts(drain(engine.events))
    assert any("No skills" in t for t in texts)


# ── List ─────────────────────────────────────────────────────────────


def test_skill_list(engine: Engine) -> None:
    """Lists all discovered skills."""
    engine.state.skill_registry = _registry(
        _skill("brave-search", "Web search."),
        _skill("hidden-skill", "Hidden.", hidden=True),
    )
    engine.run_task(TaskType.COMMAND, "/skill")
    texts = output_texts(drain(engine.events))
    assert any("brave-search" in t for t in texts)
    assert any("hidden-skill" in t for t in texts)


def test_skill_list_shows_hidden_marker(engine: Engine) -> None:
    """Hidden skills are annotated in the listing."""
    engine.state.skill_registry = _registry(
        _skill("hidden-skill", "Hidden.", hidden=True),
    )
    engine.run_task(TaskType.COMMAND, "/skill")
    texts = output_texts(drain(engine.events))
    assert any("hidden" in t and "(hidden)" in t for t in texts)


# ── Load ─────────────────────────────────────────────────────────────


@pytest.fixture
def skill_file(tmp_path: Path) -> Path:
    """A single SKILL.md on disk for load tests."""
    f = tmp_path / "SKILL.md"
    f.write_text("# Brave Search\n\nRun `./search.js` to search.\n")
    return f


def test_skill_load(engine: Engine, skill_file: Path) -> None:
    """Loading a skill emits AutoSubmit with XML-wrapped content."""
    engine.state.skill_registry = _registry(
        _skill("brave-search", "Web search.", file_path=str(skill_file)),
    )
    engine.run_task(TaskType.COMMAND, "/skill brave-search")
    events = drain(engine.events)
    assert not has_event_type(events, ContextMarkerReady)
    assert has_event_type(events, AutoSubmit)
    submit = next(e for e in events if isinstance(e, AutoSubmit))
    assert '<skill name="brave-search"' in submit.message
    assert "search.js" in submit.message
    assert "</skill>" in submit.message


def test_skill_load_strips_frontmatter(engine: Engine, tmp_path: Path) -> None:
    """Frontmatter is stripped from the skill content."""
    f = tmp_path / "SKILL.md"
    f.write_text("---\nname: test-skill\ndescription: Test.\n---\n# Body\nContent\n")
    engine.state.skill_registry = _registry(
        _skill("test-skill", "Test.", file_path=str(f)),
    )
    engine.run_task(TaskType.COMMAND, "/skill test-skill")
    events = drain(engine.events)
    submit = next(e for e in events if isinstance(e, AutoSubmit))
    # Frontmatter not inside the skill block.
    assert "---" not in submit.message.split("</skill>")[0]
    assert "# Body" in submit.message
    assert "Content" in submit.message


def test_skill_load_with_args(engine: Engine, skill_file: Path) -> None:
    """Extra args are appended after the XML block."""
    engine.state.skill_registry = _registry(
        _skill("brave-search", "Web search.", file_path=str(skill_file)),
    )
    engine.run_task(TaskType.COMMAND, "/skill brave-search pydantic ai docs")
    events = drain(engine.events)
    submit = next(e for e in events if isinstance(e, AutoSubmit))
    assert submit.message.endswith("</skill>\n\npydantic ai docs")


def test_skill_load_xml_has_location(engine: Engine, skill_file: Path) -> None:
    """XML wrapper includes the file location."""
    engine.state.skill_registry = _registry(
        _skill("brave-search", "Web search.", file_path=str(skill_file)),
    )
    engine.run_task(TaskType.COMMAND, "/skill brave-search")
    events = drain(engine.events)
    submit = next(e for e in events if isinstance(e, AutoSubmit))
    assert f'location="{skill_file}"' in submit.message


def test_skill_load_xml_has_base_dir(engine: Engine, skill_file: Path) -> None:
    """XML wrapper includes a references-relative-to line."""
    engine.state.skill_registry = _registry(
        _skill("brave-search", "Web search.", file_path=str(skill_file)),
    )
    engine.run_task(TaskType.COMMAND, "/skill brave-search")
    events = drain(engine.events)
    submit = next(e for e in events if isinstance(e, AutoSubmit))
    assert "References are relative to" in submit.message


def test_skill_load_unknown(engine: Engine) -> None:
    """Unknown skill name warns the user."""
    engine.state.skill_registry = _registry(
        _skill("brave-search", "Web search."),
    )
    engine.run_task(TaskType.COMMAND, "/skill nonexistent")
    texts = output_texts(drain(engine.events))
    assert any("Unknown skill" in t for t in texts)


def test_skill_load_hidden(engine: Engine, skill_file: Path) -> None:
    """Hidden skills can still be loaded via `/skill`."""
    engine.state.skill_registry = _registry(
        _skill("hidden-skill", "Hidden.", file_path=str(skill_file), hidden=True),
    )
    engine.run_task(TaskType.COMMAND, "/skill hidden-skill")
    events = drain(engine.events)
    assert has_event_type(events, AutoSubmit)
