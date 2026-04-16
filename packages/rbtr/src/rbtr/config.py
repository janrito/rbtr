"""Configuration for rbtr.

Sources are loaded in order (each overrides the previous):

1. Class defaults — field defaults on the model below
2. User settings  — `~/.rbtr/config.toml`
3. Environment    — `RBTR_` prefix (e.g. `RBTR_CHUNK_LINES=100`)

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

_DEFAULT_USER_DIR = str(Path.home() / ".rbtr")


class Config(BaseSettings):
    """rbtr configuration — defaults, TOML, env vars."""

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
    embed_idle_timeout: Annotated[
        int,
        Field(
            description=(
                "Seconds before unloading the embedding model when idle. "
                "Set to 0 to keep the model loaded. Only used by the daemon."            ),
        ),
    ] = 300

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
        toml = user_dir / "config.toml"
        return (
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=toml if toml.exists() else [],
            ),
            env_settings,
        )

    def reload(self) -> None:
        """Re-read all sources (env vars, TOML file) in place."""
        self.__init__()  # type: ignore[misc]  # pydantic re-init


config = Config()
