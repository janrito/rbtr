"""Layered configuration for rbtr.

Sources are loaded in order (each overrides the previous via deep merge):

1. Class defaults  — field defaults on the models below
2. User settings   — `~/.config/rbtr/config.toml`
3. Workspace       — `.rbtr/config.toml` (relative to CWD)

The `config` instance reloads in place via `reload()`, so a direct
import is safe — identity never changes::

    from rbtr.config import config

    config.model                  # read
    config.tui.poll_interval      # read nested
    config.update(model=…)        # write, persist, reload
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import tomli_w
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from rbtr import workspace  # module import so tests can patch workspace.workspace_dir once

_DEFAULT_USER_DIR = str(Path.home() / ".config" / "rbtr")

# ── Enums ────────────────────────────────────────────────────────────


class ThinkingEffort(StrEnum):
    """Thinking effort levels for LLM requests.

    Maps to provider-specific settings (`anthropic_effort`,
    `openai_reasoning_effort`, etc.) in `providers.__init__`.
    `NONE` disables thinking/reasoning entirely.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
    NONE = "none"


# ── Section models ───────────────────────────────────────────────────


class EndpointConfig(BaseModel):
    base_url: Annotated[str, Field(description="Base URL for the endpoint.")] = ""


class GithubConfig(BaseModel):
    timeout: Annotated[int, Field(description="HTTP timeout in seconds.")] = 10
    max_branches: Annotated[int, Field(description="Max branches shown in listings.")] = 30


class IndexConfig(BaseModel):
    enabled: Annotated[bool, Field(description="Enable code indexing.")] = True
    db_dir: Annotated[
        str, Field(description="Directory for the DuckDB index. Supports `${WORKSPACE}`.")
    ] = "${WORKSPACE}/index"
    max_file_size: Annotated[int, Field(description="Skip files larger than this (bytes).")] = (
        512 * 1024
    )
    include: Annotated[
        list[str], Field(description="Extra glob patterns to index (outside git tree).")
    ] = [".rbtr/notes/*", ".rbtr/AGENTS.md"]
    extend_exclude: Annotated[
        list[str], Field(description="Glob patterns to exclude from indexing.")
    ] = [".rbtr/"]
    chunk_lines: Annotated[int, Field(description="Lines per index chunk.")] = 50
    chunk_overlap: Annotated[int, Field(description="Overlap lines between adjacent chunks.")] = 5
    embedding_model: Annotated[str, Field(description="HuggingFace model ID for embeddings.")] = (
        "gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf"
    )
    embedding_batch_size: Annotated[
        int, Field(description="Batch size for embedding inference.")
    ] = 32


class MemoryConfig(BaseModel):
    enabled: Annotated[
        bool, Field(description="Enable long-term memory (fact extraction and injection).")
    ] = True
    max_injected_facts: Annotated[
        int, Field(description="Max facts injected into the system prompt.")
    ] = 20
    max_injected_tokens: Annotated[int, Field(description="Token budget for injected facts.")] = (
        2000
    )
    max_extraction_facts: Annotated[
        int, Field(description="Cap on existing facts shown to the extraction agent for dedup.")
    ] = 200
    fact_extraction_model: Annotated[
        str, Field(description="Model for fact extraction. Empty string uses the session model.")
    ] = ""


class SessionsConfig(BaseModel):
    pass


class CompactionConfig(BaseModel):
    auto_compact_pct: Annotated[
        int, Field(description="Trigger auto-compaction when context usage exceeds this %.")
    ] = 85
    keep_turns: Annotated[
        int, Field(description="Number of recent user→assistant turns to preserve.")
    ] = 2
    reserve_tokens: Annotated[
        int, Field(description="Tokens reserved for the summary response.")
    ] = 16_000
    summary_max_chars: Annotated[
        int, Field(description="Max chars per tool result in the serialised summary input.")
    ] = 2_000


class SkillsConfig(BaseModel):
    project_dirs: Annotated[
        list[str],
        Field(description="Relative dirs checked in each ancestor from CWD to project root."),
    ] = [
        ".rbtr/skills",
        ".claude/skills",
        ".pi/skills",
        ".agents/skills",
    ]
    user_dirs: Annotated[
        list[str], Field(description="Absolute paths (tilde-expanded) for user-level skills.")
    ] = [
        "~/.config/rbtr/skills",
        "~/.claude/skills",
        "~/.pi/agent/skills",
        "~/.agents/skills",
    ]
    extra_dirs: Annotated[list[str], Field(description="Additional directories to scan.")] = []


class ShellConfig(BaseModel):
    enabled: Annotated[
        bool, Field(description="Whether `run_command` is available to the LLM.")
    ] = True
    timeout: Annotated[int, Field(description="Default timeout in seconds (0 = no limit).")] = 120
    max_output_lines: Annotated[int, Field(description="Truncate output to this many lines.")] = (
        2000
    )


class ToolsConfig(BaseModel):
    shell: ShellConfig = ShellConfig()
    max_lines: Annotated[
        int, Field(description="Hard line cap for read_file, read_symbol, diff output.")
    ] = 2000
    max_results: Annotated[
        int, Field(description="Hard entry cap for list/search/reference tools.")
    ] = 50
    max_grep_hits: Annotated[int, Field(description="Max match groups returned by grep.")] = 50
    grep_context_lines: Annotated[
        int, Field(description="Lines of context above and below each grep match.")
    ] = 5
    max_search_hits: Annotated[int, Field(description="Max results returned by search.")] = 50
    search_context_lines: Annotated[
        int, Field(description="Opening lines of each symbol to preview in search results.")
    ] = 5
    notes_dir: Annotated[
        str, Field(description="Default directory for review notes. Supports `${WORKSPACE}`.")
    ] = "${WORKSPACE}/notes"
    editable_include: Annotated[
        list[str], Field(description="Glob patterns for files the `edit` tool may write.")
    ] = [".rbtr/notes/*", ".rbtr/AGENTS.md"]
    drafts_dir: Annotated[
        str, Field(description="Directory for review draft YAML files. Supports `${WORKSPACE}`.")
    ] = "${WORKSPACE}/drafts"
    max_requests_per_turn: Annotated[int, Field(description="Max tool calls per LLM turn.")] = 25
    turn_timeout: Annotated[
        float, Field(description="Max seconds for a single LLM turn. 0 = no limit.")
    ] = 300.0


class LogConfig(BaseModel):
    level: Annotated[str, Field(description="Log level (DEBUG, INFO, WARNING, ERROR).")] = "INFO"
    max_bytes: Annotated[int, Field(description="Max log file size before rotation.")] = (
        5 * 1024 * 1024
    )
    backup_count: Annotated[int, Field(description="Number of rotated log files to keep.")] = 3


class TuiConfig(BaseModel):
    shell_max_lines: Annotated[int, Field(description="Max lines shown for `!` shell output.")] = 25
    shell_context_max_chars: Annotated[
        int, Field(description="Max chars of shell output in a context marker.")
    ] = 4_000
    tool_max_lines: Annotated[int, Field(description="Max lines shown for tool results.")] = 15
    tool_max_chars: Annotated[int, Field(description="Max chars for tool results.")] = 8_000
    max_completions: Annotated[int, Field(description="Max tab-completion suggestions.")] = 20
    max_history: Annotated[
        int, Field(description="Max input history entries for Up/Down navigation.")
    ] = 500
    paste_collapse_lines: Annotated[
        int, Field(description="Pastes with this many+ lines are collapsed to a marker.")
    ] = 4
    paste_collapse_chars: Annotated[
        int, Field(description="Single-line pastes longer than this are collapsed.")
    ] = 200
    shell_completion_timeout: Annotated[
        float, Field(description="Timeout for shell tab-completion in seconds.")
    ] = 2.0
    double_ctrl_c_window: Annotated[
        float, Field(description="Seconds between Ctrl+C presses to force-quit.")
    ] = 0.5
    poll_interval: Annotated[float, Field(description="Event queue poll interval in seconds.")] = (
        1 / 30
    )
    refresh_per_second: Annotated[int, Field(description="Rich Live refresh rate.")] = 30


class OAuthConfig(BaseModel):
    refresh_buffer_seconds: Annotated[
        int, Field(description="Refresh tokens this many seconds before expiry.")
    ] = 300


# ── Theme / palette ──────────────────────────────────────────────────


class PaletteConfig(BaseModel):
    """Base palette — shared text-style defaults, mode-specific fields
    required (no defaults).

    Every field accepts any Rich style string (ANSI names, hex,
    `rgb()`, or combinations like `"bold #FF0000 on #1A1A2E"`).
    Subclass to provide defaults for backgrounds and any text styles
    that need to differ per mode.
    `styles.py` maps fields to `rbtr.*` Rich theme keys.
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
    context_marker: str = "dim cyan"
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

    `mode` selects which palette is active.
    `[theme.dark]` / `[theme.light]` override individual
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

    model_config = SettingsConfigDict(env_prefix="RBTR_")

    user_dir: Annotated[
        str,
        Field(
            exclude=True,
            description="User-level storage root. Override with `RBTR_USER_DIR` env var.",
        ),
    ] = _DEFAULT_USER_DIR
    model: Annotated[str | None, Field(description="Default model in `provider/model` format.")] = (
        None
    )
    thinking_effort: Annotated[
        ThinkingEffort, Field(description="Thinking effort level for LLM requests.")
    ] = ThinkingEffort.MEDIUM
    system_prompt_override: Annotated[
        str,
        Field(
            description="Filename in `user_dir` that replaces the built-in system prompt. Empty to disable."
        ),
    ] = "SYSTEM.md"
    append_system: Annotated[
        str,
        Field(
            description="Filename in `user_dir` appended to the system prompt. Empty to disable."
        ),
    ] = "APPEND_SYSTEM.md"
    project_instructions: Annotated[
        list[str],
        Field(
            description="Filenames read relative to repo root and injected into the system prompt."
        ),
    ] = ["AGENTS.md", ".rbtr/AGENTS.md"]
    compaction: CompactionConfig = CompactionConfig()
    endpoints: Annotated[
        dict[str, EndpointConfig], Field(description="Custom OpenAI-compatible endpoints.")
    ] = {}
    github: GithubConfig = GithubConfig()
    index: IndexConfig = IndexConfig()
    memory: MemoryConfig = MemoryConfig()
    sessions: SessionsConfig = SessionsConfig()
    skills: SkillsConfig = SkillsConfig()
    log: LogConfig = LogConfig()
    oauth: OAuthConfig = OAuthConfig()
    theme: ThemeConfig = ThemeConfig()
    tools: ToolsConfig = ToolsConfig()
    tui: TuiConfig = TuiConfig()


# ── Rendered config (merged) ─────────────────────────────────────────


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *updates* into *base*.

    An empty dict in *updates* replaces the base value (rather than
    being a no-op), so `endpoints={}` clears the section.
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
        user_dir = Path(os.environ.get("RBTR_USER_DIR", _DEFAULT_USER_DIR))
        return (
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=_toml_file(
                    user_dir / "config.toml", workspace.workspace_dir() / "config.toml"
                ),
                deep_merge=True,
            ),
            env_settings,
        )

    def update(self, **kwargs: Any) -> None:
        """Persist *kwargs* and reload.

        Saves to workspace config if it exists, otherwise to user config.
        """

        user_config = Path(self.user_dir) / "config.toml"
        workspace_config = workspace.workspace_dir() / "config.toml"
        path = workspace_config if workspace_config.exists() else user_config

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

        self.reload()

    def reload(self) -> None:
        """Re-read all sources (env vars, TOML files) in place.

        Calls `__init__()` so the singleton identity is preserved —
        all modules that imported `config` keep a valid reference.
        """
        self.__init__()  # type: ignore[misc]  # pydantic re-init


config = RenderedConfig()
