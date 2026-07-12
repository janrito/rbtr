"""Cases for per-language sample extraction tests.

Two tag-separated families share this file:

- `sample` — one case per language, returning `(lang, expected_kinds)`.
  The fixture loads the committed `samples/<lang>/` mini-project (one or
  more files) via `load_project` and feeds both the chunk and edge
  snapshots.
- `unsupported` — constructs that *should* extract but do not yet;
  each is `xfail(strict=True)` and returns `(lang, snippet,
  expected_symbol)`. Populated from Phase 2 onward.
- `sql_dialect` — one case per SQL dialect (postgres/mysql/sqlite/
  duckdb/clickhouse), returning `(source, dialect_id)`. Documents how
  the single generic SQL grammar handles each dialect today; see
  `TODO-construct-coverage.md` Appendix O.
"""

from __future__ import annotations

import pytest
from pytest_cases import case

from rbtr.index.models import ChunkKind

from .conftest import load_sample

type SampleCase = tuple[str, set[ChunkKind]]
type UnsupportedCase = tuple[str, str, tuple[ChunkKind, str, str]]
type SqlDialectCase = tuple[str, str]


@case(id="sql_postgres", tags=["sql_dialect"])
def case_sql_postgres() -> SqlDialectCase:
    """PostgreSQL: SERIAL, JSONB, arrays, ENUM type, dollar-quoted fn."""
    return load_sample("sql_postgres"), "sql_postgres"


@case(id="sql_mysql", tags=["sql_dialect"])
def case_sql_mysql() -> SqlDialectCase:
    """MySQL: backtick identifiers, UNSIGNED AUTO_INCREMENT, ENGINE."""
    return load_sample("sql_mysql"), "sql_mysql"


@case(id="sql_sqlite", tags=["sql_dialect"])
def case_sql_sqlite() -> SqlDialectCase:
    """SQLite: AUTOINCREMENT, WITHOUT ROWID, IF NOT EXISTS."""
    return load_sample("sql_sqlite"), "sql_sqlite"


@case(id="sql_duckdb", tags=["sql_dialect"])
def case_sql_duckdb() -> SqlDialectCase:
    """DuckDB: LIST/STRUCT types, CTAS, CREATE MACRO."""
    return load_sample("sql_duckdb"), "sql_duckdb"


@case(id="sql_clickhouse", tags=["sql_dialect"])
def case_sql_clickhouse() -> SqlDialectCase:
    """ClickHouse: MergeTree engine, ORDER BY/PARTITION BY, special types."""
    return load_sample("sql_clickhouse"), "sql_clickhouse"


@case(id="go", tags=["sample"])
def case_go() -> SampleCase:
    """Go: functions, methods (functions with a receiver, scoped to the
    receiver type via @_scope), struct and interface types (as classes),
    const/var (as variables), and grouped imports (one chunk per spec).
    """
    return (
        "go",
        {
            ChunkKind.FUNCTION,
            ChunkKind.METHOD,
            ChunkKind.CLASS,
            ChunkKind.VARIABLE,
            ChunkKind.IMPORT,
        },
    )


@case(id="yaml", tags=["sample"])
def case_yaml() -> SampleCase:
    """YAML: each top-level mapping key becomes a doc section (nested keys
    are part of their parent's content, not separate chunks).
    """
    return (
        "yaml",
        {ChunkKind.DOC_SECTION},
    )


@case(id="toml", tags=["sample"])
def case_toml() -> SampleCase:
    """TOML: standard tables and array-of-tables as doc sections; a dotted
    table is named by its last segment and scoped under the preceding ones
    (`[tool.greeter]` -> name `greeter`, scope `tool`).
    """
    return (
        "toml",
        {ChunkKind.DOC_SECTION},
    )


@case(id="hcl", tags=["sample"])
def case_hcl() -> SampleCase:
    """HCL: each top-level block is a doc section, named by its type and
    labels (`resource "aws_instance" "greeter"` -> `resource aws_instance
    greeter`; a bare `terraform {}` block by its type alone).
    """
    return (
        "hcl",
        {ChunkKind.DOC_SECTION},
    )


@case(id="json", tags=["sample"])
def case_json() -> SampleCase:
    """JSON: every object key becomes a doc section, including nested keys
    (the query matches all pairs; keys are flat, with no scope).
    """
    return (
        "json",
        {ChunkKind.DOC_SECTION},
    )


@case(id="css", tags=["sample"])
def case_css() -> SampleCase:
    """CSS: rule sets, @media and @charset (as doc sections; @media and
    @charset are named `<anonymous>`), @import statements (as imports), and
    custom properties (`--name`, as variables).
    """
    return (
        "css",
        {ChunkKind.DOC_SECTION, ChunkKind.IMPORT, ChunkKind.VARIABLE},
    )


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


@case(id="java", tags=["sample"])
def case_java() -> SampleCase:
    """Java: classes (incl. nested), methods (incl. constructors), fields
    and enum constants (as variables scoped to the enum), and imports (incl.
    static). No top-level functions.
    """
    return (
        "java",
        {ChunkKind.CLASS, ChunkKind.METHOD, ChunkKind.VARIABLE, ChunkKind.IMPORT},
    )


@case(id="rust", tags=["sample"])
def case_rust() -> SampleCase:
    """Rust: functions, struct/enum/impl (all as classes — a struct and
    its impl both yield one), methods inside impl (scoped to the type),
    enum variants and const/static (as variables; variants scoped to the
    enum), and `use` imports incl. `super::` dots.
    """
    return (
        "rust",
        {
            ChunkKind.FUNCTION,
            ChunkKind.CLASS,
            ChunkKind.METHOD,
            ChunkKind.VARIABLE,
            ChunkKind.IMPORT,
        },
    )


@case(
    tags=["unsupported"],
    marks=pytest.mark.xfail(
        strict=True,
        reason="tree-sitter-sql has no create_procedure node — parses to ERROR",
    ),
)
def case_sql_procedure() -> UnsupportedCase:
    """A SQL stored procedure should ideally be captured as a function."""
    src = "CREATE PROCEDURE refresh()\nLANGUAGE SQL\nAS $$ DELETE FROM cache; $$;\n"
    return "sql", src, (ChunkKind.FUNCTION, "refresh", "")


# Note: SQL PRAGMA negative space stays as the absence test
# `test_sql_pragma_not_extracted` in test_extraction.py — a `== []`
# assertion is a better sentinel than a speculative xfail tuple, since
# what a PRAGMA *should* extract is ill-defined (see plan Q8 / H11).


@case(id="c", tags=["sample"])
def case_c() -> SampleCase:
    """C: functions, struct/enum/typedef types (as classes), top-level
    variables, enum constants (as file-scope variables — C enum constants
    leak into the enclosing scope), and system/local #include imports. A
    `typedef struct G G;` yields two class chunks (the struct definition and
    the typedef alias); type references no longer produce spurious class
    chunks (plan H6).
    """
    return (
        "c",
        {ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.VARIABLE, ChunkKind.IMPORT},
    )


@case(id="cpp", tags=["sample"])
def case_cpp() -> SampleCase:
    """C++: free functions, classes, member functions and constructors
    (as methods), namespace/top-level variables, and #include imports.
    Scope uses `::` and includes the enclosing namespace.
    """
    return (
        "cpp",
        {
            ChunkKind.FUNCTION,
            ChunkKind.CLASS,
            ChunkKind.METHOD,
            ChunkKind.VARIABLE,
            ChunkKind.IMPORT,
        },
    )
