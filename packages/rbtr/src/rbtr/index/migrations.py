"""Schema migrations: what opening an index file does to bring it to this version.

`plan` decides; `Store` acts.  ARCHITECTURE's "Migrations" describes the
chains and their naming.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from enum import Enum, StrEnum

import duckdb

from rbtr.errors import IndexSchemaTooNewError, RbtrError
from rbtr.index import load_sql

_SET_SCHEMA_VERSION_SQL = load_sql("set_schema_version.sql")
_CHAINS = importlib.resources.files("rbtr.index") / "sql" / "migrations"

type Calver = tuple[int, int, int] | tuple[int, int, int, int]


def parse_calver(version: str) -> Calver:
    """Parse `YYYY.M.N` or `YYYY.M.N.P` into comparable ints; `ValueError` otherwise."""
    match [int(part) for part in version.split(".")]:
        case [year, month, release]:
            return (year, month, release)
        case [year, month, release, patch]:
            return (year, month, release, patch)
    msg = f"Not a calver version: {version!r}"
    raise ValueError(msg)


class FileKind(StrEnum):
    """A kind of index file, each with its own schema and chain of migrations."""

    REGISTRY = "registry"
    PROJECT = "project"


class Migrate(Enum):
    """When the open runs the project-file migrations that work on a copy."""

    ON_OPEN = "on_open"
    LATER = "later"  # the owner calls `Index.migrate`; until then those files refuse reads


@dataclass(frozen=True)
class Migration:
    """One migration file: the SQL for a file at any version from `start` up to `end`."""

    start: str
    end: str
    sql: str


@dataclass(frozen=True)
class Current:
    """The file is at this schema version; opening it changes nothing."""


@dataclass(frozen=True)
class Replace:
    """The file is too old to migrate, so the index is replaced; `stored` is what it recorded."""

    stored: str | None


@dataclass(frozen=True)
class BringForward:
    """The migrations that take the file from `stored` to this version; none to only record it."""

    stored: str
    migrations: tuple[Migration, ...]


type Step = Current | Replace | BringForward


def chain(kind: FileKind) -> list[Migration]:
    """The migrations of *kind*, oldest first, from its `<start>__<end>.sql` files."""
    directory = _CHAINS / kind
    if not directory.is_dir():
        return []
    migrations = []
    for entry in directory.iterdir():
        if not entry.name.endswith(".sql") or entry.name.count("__") != 1:
            continue
        start, end = entry.name.removesuffix(".sql").split("__")
        migrations.append(Migration(start=start, end=end, sql=entry.read_text(encoding="utf-8")))
    return sorted(migrations, key=lambda migration: parse_calver(migration.end))


def plan(kind: FileKind, stored: str | None, *, current: str, writable: bool) -> Step:
    """What opening a file of *kind* that records *stored* does to bring it to *current*.

    A file recording no version, one not calver, or one older than the
    oldest `start` across every kind's migrations (*current* when none
    ship) is replaced.  Raises `IndexSchemaTooNewError` for a newer file,
    and `RbtrError` naming `rbtr watch` when a read-only open would change
    a file.
    """
    try:
        version = parse_calver(stored) if stored is not None else None
    except ValueError:
        version = None
    if version is not None and version == parse_calver(current):
        return Current()
    if version is not None and version > parse_calver(current):
        raise IndexSchemaTooNewError(stored=str(stored), code=current)
    if not writable:
        msg = (
            f"Index schema {stored or 'unknown'} predates this rbtr ({current}); "
            "run `rbtr watch` to bring it forward."
        )
        raise RbtrError(msg)
    oldest = min(
        (parse_calver(shipped[0].start) for each in FileKind if (shipped := chain(each))),
        default=parse_calver(current),
    )
    if version is None or version < oldest:
        return Replace(stored=stored)
    migrations = chain(kind)
    return BringForward(
        stored=str(stored),
        migrations=tuple(m for m in migrations if parse_calver(m.end) > version),
    )


def run(cur: duckdb.DuckDBPyConnection, step: BringForward, *, current: str) -> None:
    """Apply *step*'s migrations to *cur*'s file in order, then record *current*.

    Each runs in its own transaction that records its `end`, so a failure
    re-raises and leaves the file at the last version reached.  *cur* must
    have no transaction open.
    """
    for migration in step.migrations:
        cur.begin()
        try:
            cur.execute(migration.sql)
            cur.execute(_SET_SCHEMA_VERSION_SQL, {"schema_version": migration.end})
        except BaseException:
            cur.rollback()
            raise
        cur.commit()
    cur.execute(_SET_SCHEMA_VERSION_SQL, {"schema_version": current})
