"""Scenarios for ``IndexStore`` schema and embedding-version recovery.

The fixture in ``test_store_versioning.py`` writes a DB to disk
in the state each case describes, then re-opens it through
``IndexStore(path)`` so the recovery code runs for real.  Cases
hold no I/O \u2014 only the description of what the DB should look
like before re-opening and what survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Chunk

from tests.index.cases_common import MATH_FUNC


@dataclass(frozen=True)
class DbState:
    """Pre-reopen state the fixture writes to disk."""

    # If None, the fixture creates the DB *without* our schema
    # (bare duckdb.connect + a dummy table).
    schema_version: str | None = ""  # "" means "leave as-is"
    embedding_version: int | None = None  # None means "leave as-is"
    embedding_model: str | None = None  # None means "leave as-is"

    # Chunks + snapshots inserted through IndexStore before the
    # mutations above happen.  Captured here so assertions can check
    # what survives.
    seeded_chunks: list[Chunk] = field(default_factory=list)
    seeded_embeddings: dict[str, list[float]] = field(default_factory=dict)

    # If True, the fixture skips IndexStore-based seeding entirely
    # and creates a bare DuckDB file with a dummy table \u2014 simulates
    # an ancient DB that predates our schema.
    create_bare_file: bool = False


@dataclass(frozen=True)
class VersioningScenario:
    before: DbState = field(default_factory=DbState)

    # Post-open expectations.
    expected_chunks_survive: bool = True
    expected_embeddings_survive: bool = True
    # If the scenario is supposed to change the model stamp, record it.
    expected_model_stamp: str | None = None

    # Optional: a config override applied before the second open.
    config_embedding_model: str | None = None


# ── Schema version paths ─────────────────────────────────────────────


def case_schema_version_mismatch_nukes_db() -> VersioningScenario:
    return VersioningScenario(
        before=DbState(
            schema_version="1",  # stale
            seeded_chunks=[MATH_FUNC],
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


def case_schema_version_match_keeps_data() -> VersioningScenario:
    return VersioningScenario(
        before=DbState(seeded_chunks=[MATH_FUNC]),
        expected_chunks_survive=True,
    )


# ── Embedding version paths ─────────────────────────────────────────


def case_embedding_version_mismatch_clears_embeddings() -> VersioningScenario:
    return VersioningScenario(
        before=DbState(
            embedding_version=0,  # stale
            seeded_chunks=[MATH_FUNC],
            seeded_embeddings={MATH_FUNC.id: [0.1, 0.2, 0.3]},
        ),
        expected_chunks_survive=True,
        expected_embeddings_survive=False,
    )


def case_embedding_version_match_keeps_embeddings() -> VersioningScenario:
    return VersioningScenario(
        before=DbState(
            seeded_chunks=[MATH_FUNC],
            seeded_embeddings={MATH_FUNC.id: [0.1, 0.2, 0.3]},
        ),
        expected_chunks_survive=True,
        expected_embeddings_survive=True,
    )


def case_embedding_model_change_clears_embeddings() -> VersioningScenario:
    return VersioningScenario(
        before=DbState(
            seeded_chunks=[MATH_FUNC],
            seeded_embeddings={MATH_FUNC.id: [0.1, 0.2, 0.3]},
        ),
        config_embedding_model="other/model.gguf",
        expected_chunks_survive=True,
        expected_embeddings_survive=False,
        expected_model_stamp="other/model.gguf",
    )
