"""Skill discovery — scan directories, parse frontmatter, build registry."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import frontmatter

from rbtr_legacy.config import SkillsConfig
from rbtr_legacy.skills.registry import Skill, SkillRegistry, SkillSource

log = logging.getLogger(__name__)

# ── Agent Skills spec validation limits ───────────────────────────────
# https://agentskills.io/specification#frontmatter-required

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


# ── Public API ───────────────────────────────────────────────────────


def load_skills(
    skills_config: SkillsConfig,
    *,
    project_root: str | None = None,
) -> SkillRegistry:
    """Discover skills from all configured directories.

    Scans project paths (CWD up to *project_root*), user paths,
    and any extra directories.  First-registered wins on name
    collision.

    Args:
        skills_config: Directory lists to scan.
        project_root: Root of the project (e.g. git root).
            CWD is used as both start and root when `None`.
    """
    registry = SkillRegistry()

    # Project-level: walk ancestors from CWD to project root.
    for directory in _project_scan_dirs(skills_config.project_dirs, project_root):
        _scan_dir(directory, SkillSource.PROJECT, registry)

    # User-level: fixed paths under ~.
    for raw in skills_config.user_dirs:
        directory = Path(raw).expanduser()
        _scan_dir(directory, SkillSource.USER, registry)

    # Extra directories from config.
    for raw in skills_config.extra_dirs:
        directory = Path(raw).expanduser().resolve()
        _scan_dir(directory, SkillSource.CONFIG, registry)

    if registry:
        log.info("Discovered %d skill(s)", len(registry))

    return registry


# ── Directory scanning ───────────────────────────────────────────────


def _project_scan_dirs(
    relative_dirs: list[str],
    project_root: str | None,
) -> list[Path]:
    """Build the list of project-level skill directories.

    Walks from CWD up to *project_root*, checking each ancestor
    for the configured relative dirs.  When *project_root* is
    `None`, only CWD is checked.
    """
    dirs: list[Path] = []
    cwd = Path.cwd().resolve()
    root = Path(project_root).resolve() if project_root else cwd

    current = cwd
    while True:
        for rel in relative_dirs:
            dirs.append(current / rel)
        if current == root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    return dirs


def _scan_dir(directory: Path, source: SkillSource, registry: SkillRegistry) -> None:
    """Scan a single skills directory and register found skills.

    Discovery rules (matching pi and the Agent Skills spec):
    - Direct `.md` files in the directory root
    - Recursive `SKILL.md` files in subdirectories
    - Skip dotfiles/dotdirs
    """
    if not directory.is_dir():
        return

    # Direct .md files in root.
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix == ".md":
            skill = _parse_skill_file(entry, source)
            if skill:
                registry.add(skill)

    # Recursive SKILL.md in subdirectories.
    _scan_subdirs(directory, source, registry)


def _scan_subdirs(directory: Path, source: SkillSource, registry: SkillRegistry) -> None:
    """Recursively scan subdirectories for `SKILL.md` files."""
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith("."):
            continue
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if skill_file.is_file():
            skill = _parse_skill_file(skill_file, source)
            if skill:
                registry.add(skill)
        # Continue recursing even if SKILL.md was found — nested
        # skills are allowed.
        _scan_subdirs(entry, source, registry)


# ── File parsing ─────────────────────────────────────────────────────


def _parse_skill_file(path: Path, source: SkillSource) -> Skill | None:
    """Parse a skill file and return a `Skill`, or `None` if invalid."""
    try:
        post = frontmatter.load(str(path))
    except Exception:
        log.warning("Failed to parse skill file: %s", path)
        return None

    meta = post.metadata
    skill_dir = str(path.parent)
    parent_dir_name = path.parent.name

    # Name: frontmatter > parent directory name.
    name = str(meta.get("name", "")) or parent_dir_name
    description = str(meta.get("description", ""))

    # Validate — warn but still load (lenient).
    _validate_name(name, parent_dir_name, path)

    if not description.strip():
        log.warning("Skill %s has no description — skipped: %s", name, path)
        return None

    if len(description) > MAX_DESCRIPTION_LENGTH:
        log.warning(
            "Skill %s description exceeds %d chars: %s",
            name,
            MAX_DESCRIPTION_LENGTH,
            path,
        )

    disable = meta.get("disable-model-invocation") is True

    return Skill(
        name=name,
        description=description,
        file_path=str(path),
        base_dir=skill_dir,
        source=source,
        disable_model_invocation=disable,
    )


def _validate_name(name: str, parent_dir_name: str, path: Path) -> None:
    """Log warnings for spec violations in the skill name."""
    if len(name) > MAX_NAME_LENGTH:
        log.warning(
            "Skill name exceeds %d chars (%d): %s",
            MAX_NAME_LENGTH,
            len(name),
            path,
        )
    if not _NAME_RE.match(name):
        log.warning(
            "Skill name %r has invalid characters (expected lowercase a-z, 0-9, hyphens): %s",
            name,
            path,
        )
    if name != parent_dir_name and path.name == "SKILL.md":
        log.warning(
            "Skill name %r does not match parent directory %r: %s",
            name,
            parent_dir_name,
            path,
        )
