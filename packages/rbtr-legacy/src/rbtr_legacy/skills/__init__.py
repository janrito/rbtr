"""Skill discovery and registry."""

from rbtr_legacy.skills.discovery import load_skills
from rbtr_legacy.skills.registry import Skill, SkillRegistry

__all__ = [
    "Skill",
    "SkillRegistry",
    "load_skills",
]
