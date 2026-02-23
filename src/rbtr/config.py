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

import base64
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import tomli_w
from pydantic import BaseModel, BeforeValidator, PlainSerializer
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


# ── Obfuscated type ─────────────────────────────────────────────────


def _deobfuscate(v: str) -> str:
    """Decode a base64-encoded string, or pass through if already decoded."""
    decoded = base64.b64decode(v).decode()
    if base64.b64encode(decoded.encode()).decode() == v:
        return decoded
    return v


def _obfuscate(v: str) -> str:
    """Re-encode a string to base64 for serialization."""
    return base64.b64encode(v.encode()).decode()


Obfuscated = Annotated[str, BeforeValidator(_deobfuscate), PlainSerializer(_obfuscate)]


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
    include: list[str] = [".rbtr/REVIEW-*"]
    extend_exclude: list[str] = [".rbtr/"]
    chunk_lines: int = 50
    chunk_overlap: int = 5
    embedding_model: str = "gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf"
    embedding_batch_size: int = 32


class SessionsConfig(BaseModel):
    max_sessions: int = 100
    """Maximum number of sessions to keep. Oldest pruned on startup."""
    max_age_days: int = 30
    """Sessions older than this many days are pruned on startup."""


class CompactionConfig(BaseModel):
    auto_compact_pct: int = 85
    """Trigger auto-compaction when context usage exceeds this %."""
    keep_turns: int = 5
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
    grep_context_lines: int = 50
    """Lines of context above and below each grep match."""
    workspace_prefix: str = "REVIEW-"
    """Filename prefix for writable files in ``.rbtr/``.

    The ``edit`` tool only allows writing files whose name starts
    with this prefix.  Review drafts use
    ``<prefix>DRAFT-<pr>.toml``.  The index ``include`` glob
    should match (default ``REVIEW-*`` catches both notes and
    drafts)."""
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
    shell_completion_timeout: float = 2.0
    double_ctrl_c_window: float = 0.5
    poll_interval: float = 1 / 30
    refresh_per_second: int = 30


class OAuthConfig(BaseModel):
    refresh_buffer_seconds: int = 300


class ClaudeProviderConfig(BaseModel):
    client_id: Obfuscated = _deobfuscate("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl")
    authorize_url: str = "https://claude.ai/oauth/authorize"
    token_url: str = "https://console.anthropic.com/v1/oauth/token"  # noqa: S105
    redirect_uri: str = "https://console.anthropic.com/oauth/code/callback"
    scopes: str = "org:create_api_key user:profile user:inference"
    default_model: str = "claude-sonnet-4-20250514"
    oauth_beta: str = "claude-code-20250219,oauth-2025-04-20"
    oauth_user_agent: str = "claude-cli/2.1.2 (external, cli)"


class ChatgptProviderConfig(BaseModel):
    client_id: Obfuscated = _deobfuscate("YXBwX0VNb2FtRUVaNzNmMENrWGFYcDdocmFubg==")
    authorize_url: str = "https://auth.openai.com/oauth/authorize"
    token_url: str = "https://auth.openai.com/oauth/token"  # noqa: S105
    redirect_uri: str = "http://localhost:1455/auth/callback"
    redirect_port: int = 1455
    scopes: str = "openid profile email offline_access"
    codex_base_url: str = "https://chatgpt.com/backend-api/codex"
    codex_client_version: str = "0.101.0"
    default_model: str = "gpt-4o"
    callback_timeout_seconds: int = 120
    jwt_claim_path: str = "https://api.openai.com/auth"


class OpenaiProviderConfig(BaseModel):
    default_model: str = "gpt-4o"


class GithubProviderConfig(BaseModel):
    client_id: Obfuscated = _deobfuscate("T3YyM2xpNE9UQ1l5bzJZTndBdWs=")
    device_code_url: str = "https://github.com/login/device/code"
    oauth_url: str = "https://github.com/login/oauth/access_token"


class ProvidersConfig(BaseModel):
    claude: ClaudeProviderConfig = ClaudeProviderConfig()
    chatgpt: ChatgptProviderConfig = ChatgptProviderConfig()
    openai: OpenaiProviderConfig = OpenaiProviderConfig()
    github: GithubProviderConfig = GithubProviderConfig()


# ── Config schema ────────────────────────────────────────────────────


def _toml_file(*paths: Path) -> list[str]:
    """Return list of paths if they exists (pydantic-settings ignores empty)."""
    return [str(p) for p in paths if p.exists()]


class Config(BaseSettings):
    """Schema and defaults — no file sources."""

    model_config = SettingsConfigDict()

    model: str | None = None
    thinking_effort: ThinkingEffort = ThinkingEffort.MEDIUM
    compaction: CompactionConfig = CompactionConfig()
    endpoints: dict[str, EndpointConfig] = {}
    github: GithubConfig = GithubConfig()
    index: IndexConfig = IndexConfig()
    sessions: SessionsConfig = SessionsConfig()
    log: LogConfig = LogConfig()
    oauth: OAuthConfig = OAuthConfig()
    providers: ProvidersConfig = ProvidersConfig()
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
