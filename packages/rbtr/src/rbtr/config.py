"""Layered configuration for rbtr.

Sources are loaded in order (each overrides the previous via deep merge):

1. Class defaults  — field defaults on the model below
2. User settings   — `~/.rbtr/config.toml`
3. Workspace       — `.rbtr/config.toml` (relative to CWD)

The `config` instance reloads in place via `reload()`, so a direct
import is safe — identity never changes::

    from rbtr.config import config

    config.chunk_lines      # read
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from rbtr import workspace  # module import so tests can patch workspace.workspace_dir once

_DEFAULT_USER_DIR = str(Path.home() / ".rbtr")


# ── Config ───────────────────────────────────────────────────────────


class Config(BaseSettings):
    """Schema and defaults — no file sources."""

    model_config = SettingsConfigDict(env_prefix="RBTR_", populate_by_name=True)

    user_dir: Annotated[
        str,
        Field(
            exclude=True,
            description="User-level storage root. Override with `RBTR_USER_DIR` env var.",
        ),
    ] = _DEFAULT_USER_DIR
    db_path: Annotated[str, Field(description="Path to the central DuckDB index file.")] = (
        "~/.rbtr/index.duckdb"
    )
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
    json_output: Annotated[
        bool,
        Field(
            alias="json",
            description="Force JSON output.",
        ),
    ] = False


# ── TOML file helpers ────────────────────────────────────────────────


def _toml_file(*candidates: Path) -> Path | list[Path]:
    """Return existing TOML files from *candidates*."""
    found = [p for p in candidates if p.exists()]
    if len(found) == 1:
        return found[0]
    return found


# ── Rendered config (merged) ─────────────────────────────────────────


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
                    user_dir / "config.toml",
                    workspace.workspace_dir() / "config.toml",
                ),
                deep_merge=True,
            ),
            env_settings,
        )

    def reload(self) -> None:
        """Re-read all sources (env vars, TOML files) in place."""
        self.__init__()  # type: ignore[misc]  # pydantic re-init


config = RenderedConfig()
