"""Layered configuration for rbtr.

Sources are loaded in order (each overrides the previous via deep merge):

1. Class defaults  — field defaults on the models below
2. User settings   — ``~/.config/rbtr/config.toml``
3. Workspace       — ``.rbtr/config.toml`` (relative to CWD)

The ``config`` instance reloads in place via ``__init__()``, so a direct
import is safe — identity never changes::

    from rbtr.config import config

    config.model                  # read
    config.tui.poll_interval      # read nested
    config.update(model=…)        # write, persist, reload
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from rbtr.constants import RBTR_DIR

# ── Paths for the config layers ──────────────────────────────────────

CONFIG_PATH = RBTR_DIR / "config.toml"
WORKSPACE_DIR = Path(".rbtr")
WORKSPACE_PATH = WORKSPACE_DIR / "config.toml"

# ── Enums ────────────────────────────────────────────────────────────


class ThinkingEffort(StrEnum):
    """Thinking effort levels for LLM requests.

    Maps to provider-specific settings (``anthropic_effort``,
    ``openai_reasoning_effort``, etc.) in ``providers.__init__``.
    ``NONE`` disables thinking/reasoning entirely.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
    NONE = "none"


# ── Section models ───────────────────────────────────────────────────


class EndpointConfig(BaseModel):
    base_url: str = ""


class GithubConfig(BaseModel):
    timeout: int = 10
    max_branches: int = 30


class IndexConfig(BaseModel):
    enabled: bool = True
    db_dir: str = str(WORKSPACE_DIR / "index")
    model_cache_dir: str = str(RBTR_DIR / "models")
    max_file_size: int = 512 * 1024  # 512 KiB
    include: list[str] = [".rbtr/notes/*", ".rbtr/AGENTS.md"]
    extend_exclude: list[str] = [".rbtr/"]
    chunk_lines: int = 50
    chunk_overlap: int = 5
    embedding_model: str = "gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf"
    embedding_batch_size: int = 32


class SessionsConfig(BaseModel):
    pass


class CompactionConfig(BaseModel):
    auto_compact_pct: int = 85
    """Trigger auto-compaction when context usage exceeds this %."""
    keep_turns: int = 2
    """Number of recent user→assistant turns to preserve."""
    reserve_tokens: int = 16_000
    """Tokens reserved for the summary response."""
    summary_max_chars: int = 2_000
    """Max chars per tool result in the serialised summary input."""


class ToolsConfig(BaseModel):
    max_lines: int = 2000
    """Hard line cap for read_file, read_symbol, diff output."""
    max_results: int = 50
    """Hard entry cap for list/search/reference tools."""
    max_grep_hits: int = 50
    """Max match groups returned by grep."""
    grep_context_lines: int = 5
    """Lines of context above and below each grep match."""
    notes_dir: str = str(WORKSPACE_DIR / "notes")
    """Default directory for review notes.

    Referenced in prompts so the LLM knows where to create notes.
    The directory itself is editable because ``editable_include``
    contains a matching glob by default."""
    editable_include: list[str] = [".rbtr/notes/*", ".rbtr/AGENTS.md"]
    """Glob patterns for files the ``edit`` tool may write.

    Uses the same glob syntax as ``IndexConfig.include``
    (``fnmatch`` + directory-prefix semantics)."""
    drafts_dir: str = str(WORKSPACE_DIR / "drafts")
    """Directory for review draft YAML files.

    Managed exclusively by the draft tools — not writable via
    the ``edit`` tool."""
    max_requests_per_turn: int = 25


class LogConfig(BaseModel):
    level: str = "INFO"
    max_bytes: int = 5 * 1024 * 1024  # 5 MB
    backup_count: int = 3


class TuiConfig(BaseModel):
    shell_max_lines: int = 25
    tool_max_lines: int = 15
    tool_max_chars: int = 8_000
    max_completions: int = 20
    max_history: int = 500
    """Max input history entries kept in memory for Up/Down navigation."""
    paste_collapse_lines: int = 4
    """Pastes with at least this many lines are collapsed to a marker."""
    paste_collapse_chars: int = 200
    """Single-line pastes longer than this are collapsed to a marker."""
    shell_completion_timeout: float = 2.0
    double_ctrl_c_window: float = 0.5
    poll_interval: float = 1 / 30
    refresh_per_second: int = 30


class OAuthConfig(BaseModel):
    refresh_buffer_seconds: int = 300


# ── Config schema ────────────────────────────────────────────────────


def _toml_file(*paths: Path) -> list[str]:
    """Return list of paths if they exists (pydantic-settings ignores empty)."""
    return [str(p) for p in paths if p.exists()]


class Config(BaseSettings):
    """Schema and defaults — no file sources."""

    model_config = SettingsConfigDict()

    model: str | None = None
    thinking_effort: ThinkingEffort = ThinkingEffort.MEDIUM
    system_prompt_override: str = "SYSTEM.md"
    """Filename in ``~/.config/rbtr/`` that replaces the built-in
    system prompt.  Empty string disables the override."""
    append_system: str = "APPEND_SYSTEM.md"
    """Filename in ``~/.config/rbtr/`` whose content is appended to
    the system prompt.  Empty string disables."""
    project_instructions: list[str] = ["AGENTS.md", ".rbtr/AGENTS.md"]
    """Filenames read relative to the repo root and injected into
    the system prompt.  Missing files are silently skipped."""
    compaction: CompactionConfig = CompactionConfig()
    endpoints: dict[str, EndpointConfig] = {}
    github: GithubConfig = GithubConfig()
    index: IndexConfig = IndexConfig()
    sessions: SessionsConfig = SessionsConfig()
    log: LogConfig = LogConfig()
    oauth: OAuthConfig = OAuthConfig()
    tools: ToolsConfig = ToolsConfig()
    tui: TuiConfig = TuiConfig()


# ── Rendered config (merged) ─────────────────────────────────────────


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *updates* into *base*.

    An empty dict in *updates* replaces the base value (rather than
    being a no-op), so ``endpoints={}`` clears the section.
    """
    merged = base.copy()
    for key, value in updates.items():
        if isinstance(value, dict) and value and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class RenderedConfig(Config):
    """Merged view: defaults → user → workspace (deep merge)."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:

        return (
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=_toml_file(CONFIG_PATH, WORKSPACE_PATH),
                deep_merge=True,
            ),
        )

    def update(self, **kwargs: Any) -> None:
        """Persist *kwargs* and reload.

        Saves to workspace config if it exists, otherwise to user config.
        """

        path = WORKSPACE_PATH if WORKSPACE_PATH.exists() else CONFIG_PATH

        # load specific config file, deep-merge kwargs on top
        data = _deep_merge(TomlConfigSettingsSource(Config, toml_file=path).toml_data, kwargs)

        layer = Config.model_validate(data)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            tomli_w.dumps(
                layer.model_dump(
                    exclude_none=True, exclude_unset=True, exclude_defaults=True, mode="json"
                )
            )
        )

        self.__init__()  # type: ignore[misc]  # reload in place via pydantic re-init


config = RenderedConfig()
