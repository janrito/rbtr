"""Skill model and registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

log = logging.getLogger(__name__)


class SkillSource(StrEnum):
    """Where a skill was discovered."""

    PROJECT = "project"
    USER = "user"
    CONFIG = "config"


@dataclass(frozen=True)
class Skill:
    """A discovered skill."""

    name: str
    description: str
    file_path: str
    base_dir: str
    source: SkillSource
    disable_model_invocation: bool = False


@dataclass
class SkillRegistry:
    """Collection of discovered skills.  First-registered wins on name collision."""

    _skills: dict[str, Skill] = field(default_factory=dict)

    def add(self, skill: Skill) -> None:
        """Register a skill.  Warns and skips on name collision."""
        if skill.name in self._skills:
            existing = self._skills[skill.name]
            log.warning(
                "Skill %r from %s ignored — already registered from %s",
                skill.name,
                skill.file_path,
                existing.file_path,
            )
            return
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name."""
        return self._skills.get(name)

    def visible(self) -> list[Skill]:
        """Skills that should appear in the prompt catalog.

        Excludes skills with `disable_model_invocation` set.
        """
        return [s for s in self._skills.values() if not s.disable_model_invocation]

    def all(self) -> list[Skill]:
        """All registered skills, including hidden ones."""
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)
