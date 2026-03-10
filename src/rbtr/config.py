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
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, Field
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


class MemoryConfig(BaseModel):
    enabled: bool = True
    max_facts_global: int = 50
    max_facts_repo: int = 100
    max_injected_facts: int = 20
    max_injected_tokens: int = 2000
    extraction_model: str = ""
    """Empty string means use the current session model."""


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
    turn_timeout: float = 300.0
    """Maximum seconds for a single LLM turn (including tool calls).

    Set to ``0`` to disable.  Default is 5 minutes."""


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


# ── Theme / palette ──────────────────────────────────────────────────


class PaletteConfig(BaseModel):
    """Base palette — shared text-style defaults, mode-specific fields
    required (no defaults).

    Every field accepts any Rich style string (ANSI names, hex,
    ``rgb()``, or combinations like ``"bold #FF0000 on #1A1A2E"``).
    Subclass to provide defaults for backgrounds and any text styles
    that need to differ per mode.
    ``styles.py`` maps fields to ``rbtr.*`` Rich theme keys.
    """

    # ── Text styles (shared defaults) ────────────────────────────
    prompt: str = "bold cyan"
    input_text: str = "bold"
    cursor: str = "reverse"
    dim: str = "dim"
    muted: str = "dim"
    error: str = "bold red"
    rule: str = "dim"
    footer: str = "dim"
    completion_selected: str = "bold cyan"
    completion_name: str = "bold blue"
    completion_desc: str = "dim"
    column_branch: str = "cyan"
    link: str = "bold cyan underline"
    paste_marker: str = "dim italic"
    usage_ok: str = "dim green"
    usage_critical: str = "dim red"
    usage_messages: str = "dim"
    out_dim: str = "dim"
    out_dim_italic: str = "dim italic"
    out_error: str = "bold red"
    out_shell_stderr: str = "red"

    # ── Mode-specific (required — subclasses provide defaults) ───
    warning: str
    usage_warning: str
    out_warning: str
    bg_input: str
    bg_active: str
    bg_succeeded: str
    bg_failed: str
    bg_queued: str
    bg_toolcall: str
    usage_uncertain: str


class DarkPalette(PaletteConfig):
    """Dark-mode palette — tinted panel backgrounds."""

    warning: str = "yellow"
    usage_warning: str = "dim yellow"
    out_warning: str = "yellow"
    bg_input: str = "on #282E3B"
    bg_active: str = "on #1C212C"
    bg_succeeded: str = "on #1A2620"
    bg_failed: str = "on #2A1D20"
    bg_queued: str = "on #1A1F29"
    bg_toolcall: str = "on #231D2F"
    usage_uncertain: str = "#282E3B"


class LightPalette(PaletteConfig):
    """Light-mode palette — pastel panel backgrounds."""

    warning: str = "dark_orange"
    usage_warning: str = "dim dark_orange"
    out_warning: str = "dark_orange"
    bg_input: str = "on #E8E8EC"
    bg_active: str = "on #E8ECF0"
    bg_succeeded: str = "on #E8F5E9"
    bg_failed: str = "on #FFEBEE"
    bg_queued: str = "on #ECEFF1"
    bg_toolcall: str = "on #F3E5F5"
    usage_uncertain: str = "#D0D0D0"


class ThemeConfig(BaseModel):
    """Theme configuration.

    ``mode`` selects which palette is active.
    ``[theme.dark]`` / ``[theme.light]`` override individual
    fields within each palette.
    """

    mode: Literal["dark", "light"] = "dark"
    dark: DarkPalette = Field(default_factory=DarkPalette)
    light: LightPalette = Field(default_factory=LightPalette)


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
    memory: MemoryConfig = MemoryConfig()
    sessions: SessionsConfig = SessionsConfig()
    log: LogConfig = LogConfig()
    oauth: OAuthConfig = OAuthConfig()
    theme: ThemeConfig = ThemeConfig()
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
