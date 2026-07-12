"""Cases for SQL sample extraction tests.

Two tag-separated families:

- `sample` — the SQL `samples/sql/` mini-project, returning
  `(lang, expected_kinds)`. The fixture loads the project and feeds both
  the chunk and edge snapshots.
- `sql_dialect` — one case per SQL dialect (postgres/mysql/sqlite/
  duckdb/clickhouse), returning `(source, dialect_id)`. Documents how the
  single generic SQL grammar handles each dialect today.
"""

from __future__ import annotations

from pytest_cases import case

from rbtr.index.models import ChunkKind
from rbtr.languages.testkit import load_sample

from .conftest import SAMPLES_DIR

type SampleCase = tuple[str, set[ChunkKind]]
type SqlDialectCase = tuple[str, str]


@case(id="sql", tags=["sample"])
def case_sql() -> SampleCase:
    """SQL: one chunk per top-level statement — CREATE TABLE/VIEW (as
    classes), CREATE FUNCTION and DML/CTEs (as functions), and CREATE
    SCHEMA/INDEX/SEQUENCE (as variables). No imports. CREATE PROCEDURE and
    PRAGMA do not parse and live in the sql xfail cases.
    """
    return (
        "sql",
        {ChunkKind.CLASS, ChunkKind.FUNCTION, ChunkKind.VARIABLE},
    )


@case(id="sql_postgres", tags=["sql_dialect"])
def case_sql_postgres() -> SqlDialectCase:
    """PostgreSQL: SERIAL, JSONB, arrays, ENUM type, dollar-quoted fn."""
    return load_sample(SAMPLES_DIR, "sql_postgres"), "sql_postgres"


@case(id="sql_mysql", tags=["sql_dialect"])
def case_sql_mysql() -> SqlDialectCase:
    """MySQL: backtick identifiers, UNSIGNED AUTO_INCREMENT, ENGINE."""
    return load_sample(SAMPLES_DIR, "sql_mysql"), "sql_mysql"


@case(id="sql_sqlite", tags=["sql_dialect"])
def case_sql_sqlite() -> SqlDialectCase:
    """SQLite: AUTOINCREMENT, WITHOUT ROWID, IF NOT EXISTS."""
    return load_sample(SAMPLES_DIR, "sql_sqlite"), "sql_sqlite"


@case(id="sql_duckdb", tags=["sql_dialect"])
def case_sql_duckdb() -> SqlDialectCase:
    """DuckDB: LIST/STRUCT types, CTAS, CREATE MACRO."""
    return load_sample(SAMPLES_DIR, "sql_duckdb"), "sql_duckdb"


@case(id="sql_clickhouse", tags=["sql_dialect"])
def case_sql_clickhouse() -> SqlDialectCase:
    """ClickHouse: MergeTree engine, ORDER BY/PARTITION BY, special types."""
    return load_sample(SAMPLES_DIR, "sql_clickhouse"), "sql_clickhouse"
