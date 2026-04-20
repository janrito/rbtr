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
from typing import Annotated, Self

from pydantic import AfterValidator, Field, computed_field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

_DEFAULT_HOME = Path.home() / ".rbtr"


class Config(BaseSettings):
    """rbtr configuration — defaults, TOML, env vars."""

    model_config = SettingsConfigDict(env_prefix="RBTR_", populate_by_name=True)

    # `AfterValidator(Path.expanduser)` runs after pydantic's
    # ``str -> Path`` coercion so ``RBTR_HOME=~/foo`` resolves
    # properly; pydantic does not expand ``~`` by itself.
    home: Annotated[
        Path,
        AfterValidator(Path.expanduser),
        Field(
            exclude=True,
            description=(
                "Base directory for all rbtr state — DB, config,"
                " sockets, logs, models.  Override with `RBTR_HOME`"
                " env var or `--home` flag."
            ),
        ),
    ] = _DEFAULT_HOME
    db_name: Annotated[
        str,
        Field(
            description=(
                "Filename for the central DuckDB index under ``home``. "
                "Override with `RBTR_DB_NAME` env var."
            ),
        ),
    ] = "index.duckdb"

    @computed_field  # type: ignore[prop-decorator]  # mypy #1362: decorated property
    @property
    def db_path(self) -> Path:
        """Full path to the DuckDB index, derived from ``home`` + ``db_name``.

        Always ``{home}/{db_name}``. A read-only computed field
        (no separate setting) so DB storage stays co-located with
        everything else under ``home`` — ``RBTR_HOME=X`` gives
        real isolation with no second knob to forget.
        """
        return self.home / self.db_name

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
                "Set to 0 to keep the model loaded. Only used by the daemon."
            ),
        ),
    ] = 300
    idle_poll_interval: Annotated[
        float,
        Field(
            description=(
                "Seconds between watcher polls while the build queue is idle. "
                "Only used by the daemon."
            ),
        ),
    ] = 5.0
    busy_poll_interval: Annotated[
        float,
        Field(
            description=(
                "Seconds between watcher polls while a build is in progress. "
                "Slowed down to avoid flooding the queue with duplicates. "
                "Only used by the daemon."
            ),
        ),
    ] = 30.0
    search_alpha: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Default fusion weight for the semantic (embedding) channel. "
                "alpha + beta + gamma must sum to 1."
            ),
        ),
    ] = 0.4
    search_beta: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Default fusion weight for the lexical (BM25) channel.",
        ),
    ] = 0.1
    search_gamma: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Default fusion weight for the name-match channel.",
        ),
    ] = 0.5

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        home = Path(os.environ.get("RBTR_HOME", _DEFAULT_HOME)).expanduser()
        toml = home / "config.toml"
        return (
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=toml if toml.exists() else [],
            ),
            env_settings,
        )

    @model_validator(mode="after")
    def _check_search_weights_sum(self) -> Self:
        total = self.search_alpha + self.search_beta + self.search_gamma
        if abs(total - 1.0) > 1e-6:
            msg = f"search_alpha + search_beta + search_gamma must sum to 1.0; got {total:.6f}"
            raise ValueError(msg)
        return self

    def reload(self) -> None:
        """Re-read all sources (env vars, TOML file) in place."""
        self.__init__()  # type: ignore[misc]  # pydantic re-init


config = Config()
