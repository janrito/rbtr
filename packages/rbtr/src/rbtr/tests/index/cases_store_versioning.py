"""Scenarios for schema and embedding-version recovery.

Cases take the `math_func` fixture from `conftest.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Chunk


@dataclass(frozen=True)
class DbState:
    """Pre-reopen state the fixture writes to disk."""

    schema_version: str | None = ""
    embedding_model: str | None = None
    embedding_version: str | None = None
    seeded_chunks: list[Chunk] = field(default_factory=list)
    seeded_embeddings: dict[str, list[float]] = field(default_factory=dict)
    create_bare_file: bool = False


@dataclass(frozen=True)
class VersioningScenario:
    before: DbState = field(default_factory=DbState)
    expected_chunks_survive: bool = True
    expected_embeddings_survive: bool = True
    expected_meta_stored: bool = False
    config_embedding_model: str | None = None


def case_schema_version_mismatch_nukes_db(math_func: Chunk) -> VersioningScenario:
    return VersioningScenario(
        before=DbState(
            schema_version="1",
            seeded_chunks=[math_func],
        ),
        expected_chunks_survive=False,
        expected_embeddings_survive=False,
    )


def case_schema_missing_meta_table_nukes_db() -> VersioningScenario:
    return VersioningScenario(
        before=DbState(create_bare_file=True),
        expected_chunks_survive=False,
        expected_embeddings_survive=False,
    )


def case_schema_version_match_keeps_data(math_func: Chunk) -> VersioningScenario:
    return VersioningScenario(
        before=DbState(seeded_chunks=[math_func]),
    )


def case_embedding_meta_mismatch_clears_embeddings(
    math_func: Chunk,
) -> VersioningScenario:
    """Stale embedding meta (model or version) triggers clearing."""
    return VersioningScenario(
        before=DbState(
            embedding_model="stale/model.gguf",
            embedding_version="999",
            seeded_chunks=[math_func],
            seeded_embeddings={math_func.id: [0.1, 0.2, 0.3]},
        ),
        expected_embeddings_survive=False,
        expected_meta_stored=True,
    )


def case_embedding_meta_match_keeps_embeddings(
    math_func: Chunk,
) -> VersioningScenario:
    """Current model and version match — embeddings survive."""
    return VersioningScenario(
        before=DbState(
            # None = let the fixture use the current values.
            embedding_model=None,
            embedding_version=None,
            seeded_chunks=[math_func],
            seeded_embeddings={math_func.id: [0.1, 0.2, 0.3]},
        ),
        expected_embeddings_survive=True,
    )


def case_embedding_model_change_clears_embeddings(
    math_func: Chunk,
) -> VersioningScenario:
    """Changing config.embedding_model clears embeddings."""
    return VersioningScenario(
        before=DbState(
            seeded_chunks=[math_func],
            seeded_embeddings={math_func.id: [0.1, 0.2, 0.3]},
        ),
        config_embedding_model="other/model.gguf",
        expected_embeddings_survive=False,
        expected_meta_stored=True,
    )
