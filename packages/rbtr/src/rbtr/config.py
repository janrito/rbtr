"""Configuration for rbtr.

Sources are loaded in order (each overrides the previous):

1. Class defaults — field defaults on the model below
2. User settings  — `{config_dir}/config.toml`
3. Environment    — `RBTR_` prefix (e.g. `RBTR_CHUNK_LINES=100`)

Four independent directory fields default to the platform-
native `platformdirs` paths; each can be overridden via its
`RBTR_*_DIR` env var or `--*-dir` CLI flag:

- `data_dir`   → `user_data_path("rbtr")`   (DuckDB index)
- `config_dir` → `user_config_path("rbtr")` (`config.toml`)
- `log_dir`    → `user_log_path("rbtr")`    (`daemon.log`)
- `cache_dir`  → `user_cache_path("rbtr")`  (embedding models)

Derived paths are exposed as computed fields:

- `runtime_dir`  — `platformdirs.user_runtime_path / hash(data_dir)`,
                   for sockets + status file
- `db_path`      — `data_dir / db_name`
- `daemon_log`   — `log_dir / "daemon.log"`
- `daemon_rpc`   — `runtime_dir / "daemon.rpc"`
- `daemon_pub`   — `runtime_dir / "daemon.pub"`

The `config` instance reloads in place via `reload()`; a
direct import is safe — identity never changes::

    from rbtr.config import config

    config.chunk_lines      # read
"""

from __future__ import annotations

import enum
import hashlib
import os
from pathlib import Path
from typing import Self

import platformdirs
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from rbtr.index.models import QueryKind

RBTR_NAME = "rbtr"


class LogFormat(enum.StrEnum):
    """Renderer selection for log output.

    `AUTO` resolves to `CONSOLE` when stderr is a TTY, else `JSON`.
    """

    AUTO = "auto"
    CONSOLE = "console"
    JSON = "json"


class RerankerSettings(BaseModel):
    """Per-query-kind reranker parameters."""

    pool: int = Field(ge=1, description="Candidates passed to reranker.")
    blend_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="score = w * fusion + (1-w) * reranker.",
    )


class WeightTriple(BaseModel):
    """Fusion channel weights (must sum to 1.0)."""

    alpha: float = Field(ge=0.0, le=1.0)
    beta: float = Field(ge=0.0, le=1.0)
    gamma: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum(self) -> Self:
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-3:
            msg = f"alpha + beta + gamma must sum to 1.0; got {total:.6f}"
            raise ValueError(msg)
        return self


class Config(BaseSettings):
    """rbtr configuration — defaults, TOML, env vars."""

    model_config = SettingsConfigDict(
        env_prefix="RBTR_",
        env_nested_delimiter="__",
        populate_by_name=True,
    )

    data_dir: Path = Field(
        default_factory=lambda: platformdirs.user_data_path(RBTR_NAME, ensure_exists=True),
        description="Directory for the DuckDB index.",
    )
    config_dir: Path = Field(
        default_factory=lambda: platformdirs.user_config_path(RBTR_NAME, ensure_exists=True),
        description="Directory for `config.toml`.",
    )
    log_dir: Path = Field(
        default_factory=lambda: platformdirs.user_log_path(RBTR_NAME, ensure_exists=True),
        description="Directory for `daemon.log`.",
    )
    cache_dir: Path = Field(
        default_factory=lambda: platformdirs.user_cache_path(RBTR_NAME, ensure_exists=True),
        description="Directory for regeneratable caches (embedding models).",
    )
    db_name: str = Field(
        default="index.duckdb",
        description="Filename of the DuckDB index under `data_dir`.",
    )
    max_file_size: int = Field(
        default=512 * 1024,
        description="Skip files larger than this (bytes).",
    )
    chunk_lines: int = Field(default=50, description="Lines per index chunk.")
    chunk_overlap: int = Field(default=5, description="Overlap lines between adjacent chunks.")
    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
        description="HuggingFace model ID for embeddings.",
    )
    query_instruction: str = Field(
        default="Instruct: Given a code search query, retrieve relevant code or documentation\nQuery:",
        description="Instruction prefix prepended to queries before embedding.  "
        "Model-specific; must match the model's training format.",
    )
    embedding_pooling_type: int = Field(
        default=3,
        description="Pooling type for the embedding model.  "
        "-1 = auto-detect from GGUF metadata.  "
        "3 = last-token pooling (required for decoder models like Qwen3-Embedding).",
    )
    embedding_n_ctx: int = Field(
        default=2048,
        description="Context window (and batch size) for the embedding model.  "
        "Must be >= the longest chunk in tokens.  Example: a 50-line "
        "Python function tokenises to ~600 tokens, so 2048 is safe.  "
        "If set too low (e.g. 512) and a chunk has 600 tokens, the "
        "embedding is SILENTLY CORRUPTED — the decoder only sees the "
        "last 512 tokens before pooling.  Larger than needed (e.g. "
        "32768) wastes ~1 GB Metal memory and slows inference ~1.3x.",
    )
    embedding_n_gpu_layers: int = Field(
        default=-1,
        description="Layers to offload to GPU.  "
        "-1 = all (full Metal/CUDA acceleration).  "
        "0 = CPU-only.",
    )
    embedding_verbose: bool = Field(
        default=False,
        description="Enable verbose llama.cpp output during model loading.",
    )
    embedding_batch_size: int = Field(default=32, description="Batch size for embedding inference.")
    insert_batch_size: int = Field(
        default=512,
        description="Max chunks buffered before a DuckDB insert flush. "
        "Caps peak memory during indexing.",
    )
    reranker_model: str | None = Field(
        default="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf",
        description="HuggingFace GGUF model ID for cross-encoder reranking.  "
        "Set to empty string or None to disable.",
    )
    reranker_settings: dict[QueryKind, RerankerSettings] = Field(
        default={
            QueryKind.CONCEPT: RerankerSettings(pool=50, blend_weight=0.25),
            QueryKind.IDENTIFIER: RerankerSettings(pool=20, blend_weight=0.5),
            QueryKind.CODE: RerankerSettings(pool=20, blend_weight=0.25),
        },
        description="Per-query-kind reranker pool size and blend weight.  "
        "Every search classifies the query and selects the matching entry.",
    )
    reranker_idle_timeout: int = Field(
        default=300,
        description="Seconds before unloading the reranker model when idle.  "
        "Set to 0 to keep the model loaded.  Only used by the daemon.",
    )
    json_output: bool = Field(default=False, alias="json", description="Force JSON output.")
    log_level: str = Field(
        default="INFO",
        description="Root log level (e.g. DEBUG, INFO, WARNING).  Raised by `-v`.",
    )
    log_format: LogFormat = Field(
        default=LogFormat.AUTO,
        description="Log renderer: auto (console on a TTY, else JSON), console, or json.",
    )
    log_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Max size of `daemon.log` before rotation (bytes).",
    )
    log_backup_count: int = Field(
        default=5,
        description="Number of rotated `daemon.log` backups to keep.",
    )
    embed_idle_timeout: int = Field(
        default=300,
        description="Seconds before unloading the embedding model when idle.  \
Set to 0 to keep the model loaded.  Only used by the daemon.",
    )
    idle_poll_interval: float = Field(
        default=5.0,
        description="Seconds between watcher polls while the build queue is idle.  \
Only used by the daemon.",
    )
    busy_poll_interval: float = Field(
        default=30.0,
        description="Seconds between watcher polls while a build is in progress.  \
Slowed down to avoid flooding the queue with duplicates.  Only used by the daemon.",
    )
    daemon_recv_timeout_ms: int = Field(
        default=30_000,
        description="ZMQ receive timeout (ms) for the daemon client.  \
Must accommodate the first search when the embedding model \
is still loading.",
    )
    daemon_start_timeout: float = Field(
        default=60.0,
        description="Backstop seconds to wait for a spawned daemon to bind "
        "its sockets before giving up.  Only trips on a genuine hang; a "
        "slow cold start under load binds well within this.",
    )
    warmup: bool = Field(
        default=True,
        description="Pre-load GPU models (embedder, reranker) when the daemon starts.  \
Disable in tests or resource-constrained environments.",
    )
    retrieval_multiplier_lexical: int = Field(
        default=5,
        description="BM25 retrieval pool = limit * this multiplier.",
    )
    retrieval_multiplier_semantic: int = Field(
        default=10,
        description="Semantic retrieval pool = limit * this multiplier.  "
        "Pre-filter; trimmed to the lexical pool size after cosine cutoff.",
    )
    max_expansion_keywords: int = Field(
        default=10,
        ge=1,
        description="Cap on a search request's expansion keywords; the excess is dropped.",
    )
    search_weights: dict[QueryKind, WeightTriple] = Field(
        default={
            QueryKind.CONCEPT: WeightTriple(alpha=0.50, beta=0.40, gamma=0.10),
            QueryKind.IDENTIFIER: WeightTriple(alpha=0.05, beta=0.20, gamma=0.75),
            QueryKind.CODE: WeightTriple(alpha=0.05, beta=0.30, gamma=0.65),
        },
        description="Per-query-kind fusion weights (semantic, lexical, name-match). "
        "Every search classifies the query and selects the matching triple.",
    )

    # `# type: ignore[prop-decorator]` on every `@computed_field`
    # below is the pydantic-docs-recommended workaround for mypy
    # issue #1362 / #14461: mypy can't follow a decorator sitting
    # on top of `@property`.  Pydantic's alternative (drop the
    # explicit `@property` and let `@computed_field` wrap the
    # function) silently breaks static type-checking on reads.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def runtime_dir(self) -> Path:
        """Per-`data_dir` runtime dir for sockets + status file.

        Keyed on `hash(resolve(data_dir))` so two daemons
        against different data dirs get independent runtime
        dirs.  Lives under `platformdirs.user_runtime_path('rbtr')`.
        """
        base = platformdirs.user_runtime_path(RBTR_NAME, ensure_exists=True)
        key = hashlib.sha256(str(self.data_dir.resolve()).encode()).hexdigest()[:16]
        path = base / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    @computed_field  # type: ignore[prop-decorator]
    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_name

    @computed_field  # type: ignore[prop-decorator]
    @property
    def daemon_log(self) -> Path:
        return self.log_dir / "daemon.log"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def daemon_rpc(self) -> Path:
        return self.runtime_dir / "daemon.rpc"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def daemon_pub(self) -> Path:
        return self.runtime_dir / "daemon.pub"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        override = os.environ.get("RBTR_CONFIG_DIR")
        config_dir = (
            Path(override).expanduser()
            if override
            else platformdirs.user_config_path(RBTR_NAME, ensure_exists=True)
        )
        toml = config_dir / "config.toml"
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=toml if toml.exists() else [],
            ),
        )

    @field_validator("data_dir", "config_dir", "log_dir", "cache_dir", mode="after")
    @classmethod
    def _expand_user_tilde(cls, v: Path) -> Path:
        return v.expanduser()

    def reload(self) -> None:
        """Re-read all sources (env vars, TOML file) in place."""
        self.__init__()  # type: ignore[misc]  # pydantic re-init


config = Config()
