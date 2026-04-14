"""Handler for `/skill` — list or load skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

import frontmatter

from rbtr.events import AutoSubmit
from rbtr.skills.registry import SkillSource

if TYPE_CHECKING:
    from .core import Engine


def cmd_skill(engine: Engine, args: str) -> None:
    """List available skills or load one by name.

    `/skill`              — list all discovered skills.
    `/skill <name>`       — load and send to the LLM.
    `/skill <name> <msg>` — load with *msg* appended.
    """
    registry = engine.state.skill_registry
    if registry is None or len(registry) == 0:
        engine._warn("No skills discovered. Check skill directories with /reload.")
        return

    if not args:
        _list_skills(engine)
        return

    parts = args.split(None, 1)
    name = parts[0]
    extra = parts[1] if len(parts) > 1 else ""

    skill = registry.get(name)
    if skill is None:
        engine._warn(f"Unknown skill: {name}")
        return

    _load_skill(engine, skill.file_path, skill.base_dir, name, extra)


def _list_skills(engine: Engine) -> None:
    """Print all discovered skills grouped by source."""
    registry = engine.state.skill_registry
    if registry is None:
        return

    by_source: dict[SkillSource, list[tuple[str, str, bool]]] = {}
    for skill in registry.all():
        by_source.setdefault(skill.source, []).append(
            (skill.name, skill.description, skill.disable_model_invocation),
        )

    for source in SkillSource:
        entries = by_source.get(source)
        if not entries:
            continue
        engine._out(f"  {source.value}:")
        for name, description, hidden in sorted(entries):
            suffix = " (hidden)" if hidden else ""
            engine._out(f"    {name:<24} {description}{suffix}")


def _load_skill(
    engine: Engine,
    file_path: str,
    base_dir: str,
    name: str,
    extra: str,
) -> None:
    """Read a skill file, wrap in XML, and auto-submit to the LLM.

    Format matches pi's `<skill>` block so the model sees
    metadata (name, location) and a clear content boundary.
    """
    try:
        with open(file_path) as f:
            raw = f.read()
    except OSError as exc:
        engine._warn(f"Failed to read skill {name}: {exc}")
        return

    body = frontmatter.loads(raw).content.strip()

    skill_block = f"""\
<skill name="{name}" location="{file_path}">
References are relative to {base_dir}.

{body}
</skill>"""

    message = f"{skill_block}\n\n{extra}" if extra else skill_block
    engine._emit(AutoSubmit(message=message))
