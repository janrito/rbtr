"""Scenarios for schema and embedding-version recovery.

Cases take the `math_func` fixture from `conftest.py`.  Cases tagged
`migration` open an index whose files record older schema versions
and return a `MigrationScenario`; the rest return a
`VersioningScenario`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
from pytest_cases import case

from rbtr.domain.models import Chunk
from rbtr.errors import IndexSchemaTooNewError, RbtrError


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
    """`migrations` maps `<kind>/<start>__<end>.sql` to the SQL this rbtr ships, at `current`."""

    before: DbState = field(default_factory=DbState)
    expected_chunks_survive: bool = True
    expected_embeddings_survive: bool = True
    expected_meta_stored: bool = False
    config_embedding_model: str | None = None
    migrations: dict[str, str] = field(default_factory=dict)
    current: str | None = None


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


def case_schema_older_than_every_shipped_migration_nukes_db(math_func: Chunk) -> VersioningScenario:
    """A retired migration leaves the files that needed it too old to migrate."""
    return VersioningScenario(
        before=DbState(schema_version="2026.10.0", seeded_chunks=[math_func]),
        migrations={"registry/2026.11.0__2026.12.0.sql": "CREATE TABLE added_note (x INT);"},
        current="2026.12.0",
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


@dataclass(frozen=True)
class FileAfter:
    """What a file records once the open is over: its schema version, and the tables a migration added."""

    version: str | None
    added: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MigrationScenario:
    """An index with one indexed chunk, its files recording older versions, opened by an rbtr at `current`.

    `migrations` maps `<kind>/<start>__<end>.sql` to its SQL; a migration
    adds a table whose name starts `added_`.  `raises` is what the open
    raises, `None` when it succeeds.  `copied` says whether the project's
    file was switched for a migrated copy.
    """

    registry_stored: str
    project_stored: str | None
    current: str
    migrations: dict[str, str]
    registry_after: FileAfter
    project_after: FileAfter
    writable: bool = True
    raises: type[Exception] | None = None
    copied: bool = False


@case(tags="migration")
def case_registry_in_place() -> MigrationScenario:
    """The registry migrates in place; a project file with no migrations records the version."""
    return MigrationScenario(
        registry_stored="2026.10.0",
        project_stored="2026.10.0",
        current="2026.11.0",
        migrations={"registry/2026.10.0__2026.11.0.sql": "CREATE TABLE added_note (x INT);"},
        registry_after=FileAfter("2026.11.0", frozenset({"added_note"})),
        project_after=FileAfter("2026.11.0"),
    )


@case(tags="migration")
def case_between_start_and_end() -> MigrationScenario:
    """A file between a migration's start and end has its start's schema, so it runs."""
    return MigrationScenario(
        registry_stored="2026.10.5",
        project_stored="2026.10.5",
        current="2026.11.0",
        migrations={"registry/2026.10.0__2026.11.0.sql": "CREATE TABLE added_note (x INT);"},
        registry_after=FileAfter("2026.11.0", frozenset({"added_note"})),
        project_after=FileAfter("2026.11.0"),
    )


@case(tags="migration")
def case_project_on_a_copy() -> MigrationScenario:
    """A project file migrations change is migrated on a copy the registry then names."""
    return MigrationScenario(
        registry_stored="2026.10.0",
        project_stored="2026.10.0",
        current="2026.11.0",
        migrations={"project/2026.10.0__2026.11.0.sql": "CREATE TABLE added_note (x INT);"},
        registry_after=FileAfter("2026.11.0"),
        project_after=FileAfter("2026.11.0", frozenset({"added_note"})),
        copied=True,
    )


@case(tags="migration")
def case_project_records_no_version() -> MigrationScenario:
    """A project file that records no version has the registry's, and a writable open records it."""
    return MigrationScenario(
        registry_stored="2026.11.0",
        project_stored=None,
        current="2026.11.0",
        migrations={},
        registry_after=FileAfter("2026.11.0"),
        project_after=FileAfter("2026.11.0"),
    )


@case(tags="migration")
def case_project_records_no_version_read_only() -> MigrationScenario:
    """A read-only open reads a project file that records no version as the registry's, unchanged."""
    return MigrationScenario(
        registry_stored="2026.11.0",
        project_stored=None,
        current="2026.11.0",
        migrations={},
        registry_after=FileAfter("2026.11.0"),
        project_after=FileAfter(None),
        writable=False,
    )


@case(tags="migration")
def case_failing_registry_stops_at_the_last_version_reached() -> MigrationScenario:
    """Each migration commits alone with the version it reaches; the failing one rolls back."""
    return MigrationScenario(
        registry_stored="2026.10.0",
        project_stored="2026.10.0",
        current="2026.12.0",
        migrations={
            "registry/2026.10.0__2026.11.0.sql": "CREATE TABLE added_note (x INT);",
            "registry/2026.11.0__2026.12.0.sql": """\
CREATE TABLE added_later (x INT);
SELECT * FROM no_such_table;
""",
        },
        registry_after=FileAfter("2026.11.0", frozenset({"added_note"})),
        project_after=FileAfter("2026.10.0"),
        raises=duckdb.CatalogException,
    )


@case(tags="migration")
def case_failing_project_keeps_the_old_file() -> MigrationScenario:
    """A failing project migration deletes its copy; the registry still names the old file."""
    return MigrationScenario(
        registry_stored="2026.10.0",
        project_stored="2026.10.0",
        current="2026.11.0",
        migrations={"project/2026.10.0__2026.11.0.sql": "SELECT * FROM no_such_table;"},
        registry_after=FileAfter("2026.11.0"),
        project_after=FileAfter("2026.10.0"),
        raises=duckdb.CatalogException,
    )


@case(tags="migration")
def case_newer_than_this_rbtr_is_refused() -> MigrationScenario:
    """An older rbtr refuses an index a newer one wrote, and leaves it for that rbtr."""
    return MigrationScenario(
        registry_stored="2026.10.0",
        project_stored="2026.10.0",
        current="2026.9.5",
        migrations={},
        registry_after=FileAfter("2026.10.0"),
        project_after=FileAfter("2026.10.0"),
        raises=IndexSchemaTooNewError,
    )


@case(tags="migration")
def case_read_only_is_refused() -> MigrationScenario:
    """Only a writable open brings a file forward; a read-only one names the command that does."""
    return MigrationScenario(
        registry_stored="2026.10.0",
        project_stored="2026.10.0",
        current="2026.11.0",
        migrations={},
        registry_after=FileAfter("2026.10.0"),
        project_after=FileAfter("2026.10.0"),
        writable=False,
        raises=RbtrError,
    )
