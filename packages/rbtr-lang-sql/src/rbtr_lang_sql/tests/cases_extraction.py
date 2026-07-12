"""SQL extraction test cases.

Each `@case` returns test data consumed by `test_extraction.py` via
`pytest-cases`. The cases are the spec of the SQL constructs the plugin
captures — one chunk per top-level statement (plus one per CTE), named
by its object/table (SQL has no nesting, so scope is always ""). See the
plugin docstring for the full source→chunk mapping.
"""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]


# ── DDL definitions: class ───────────────────────────────────────────


@case(tags=["symbol"])
def case_sql_table() -> SymbolCase:
    """CREATE TABLE — structural definition → class."""
    return "sql", "CREATE TABLE users (id INT, name TEXT);\n", [("class", "users", "")]


@case(tags=["symbol"])
def case_sql_view() -> SymbolCase:
    """CREATE VIEW → class."""
    return "sql", "CREATE VIEW active AS SELECT * FROM users;\n", [("class", "active", "")]


@case(tags=["symbol"])
def case_sql_materialized_view() -> SymbolCase:
    """CREATE MATERIALIZED VIEW → class."""
    src = "CREATE MATERIALIZED VIEW recent AS SELECT * FROM users;\n"
    return "sql", src, [("class", "recent", "")]


@case(tags=["symbol"])
def case_sql_type_enum() -> SymbolCase:
    """CREATE TYPE ... AS ENUM — SQL's enum → class."""
    return "sql", "CREATE TYPE mood AS ENUM ('sad', 'happy');\n", [("class", "mood", "")]


@case(tags=["symbol"])
def case_sql_type_composite() -> SymbolCase:
    """CREATE TYPE ... AS (...) — composite type → class."""
    src = "CREATE TYPE point AS (x DOUBLE PRECISION, y DOUBLE PRECISION);\n"
    return "sql", src, [("class", "point", "")]


# ── DDL definitions: variable (standalone named objects) ─────────────


@case(tags=["symbol"])
def case_sql_index() -> SymbolCase:
    """CREATE INDEX → variable, named by the index (not the ON table)."""
    return "sql", "CREATE INDEX idx_name ON users (name);\n", [("variable", "idx_name", "")]


@case(tags=["symbol"])
def case_sql_sequence() -> SymbolCase:
    """CREATE SEQUENCE → variable."""
    return "sql", "CREATE SEQUENCE order_id START 1;\n", [("variable", "order_id", "")]


@case(tags=["symbol"])
def case_sql_schema() -> SymbolCase:
    """CREATE SCHEMA → variable."""
    return "sql", "CREATE SCHEMA app;\n", [("variable", "app", "")]


@case(tags=["symbol"])
def case_sql_extension() -> SymbolCase:
    """CREATE EXTENSION → variable (common in migrations)."""
    return "sql", "CREATE EXTENSION postgis;\n", [("variable", "postgis", "")]


@case(tags=["symbol"])
def case_sql_trigger() -> SymbolCase:
    """CREATE TRIGGER → variable, named by the trigger (not its table/function)."""
    src = """\
CREATE TRIGGER audit BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION log_change();
"""
    return "sql", src, [("variable", "audit", "")]


# ── Routines and statements: function ────────────────────────────────


@case(tags=["symbol"])
def case_sql_function() -> SymbolCase:
    """CREATE FUNCTION — routine → function."""
    src = """\
CREATE FUNCTION add(a INT, b INT) RETURNS INT
LANGUAGE SQL
AS $$ SELECT a + b; $$;
"""
    return "sql", src, [("function", "add", "")]


@case(tags=["symbol"])
def case_sql_select() -> SymbolCase:
    """SELECT → function, named by its primary FROM table."""
    return "sql", "SELECT id, name FROM users;\n", [("function", "users", "")]


@case(tags=["symbol"])
def case_sql_insert() -> SymbolCase:
    """INSERT → function, named by its target table."""
    return "sql", "INSERT INTO logs (msg) VALUES ('hi');\n", [("function", "logs", "")]


@case(tags=["symbol"])
def case_sql_update() -> SymbolCase:
    """UPDATE → function, named by its target table."""
    return "sql", "UPDATE users SET name = 'x' WHERE id = 1;\n", [("function", "users", "")]


@case(tags=["symbol"])
def case_sql_delete() -> SymbolCase:
    """DELETE → function, named by its target table."""
    return "sql", "DELETE FROM sessions WHERE id = 1;\n", [("function", "sessions", "")]


@case(tags=["symbol"])
def case_sql_cte() -> SymbolCase:
    """Each CTE → a function named by its identifier."""
    src = """\
WITH ranked AS (
    SELECT id FROM events
),
recent AS (
    SELECT id FROM ranked
)
SELECT id FROM recent;
"""
    return "sql", src, [("function", "ranked", ""), ("function", "recent", "")]


# ── DDL operations: function (alter/drop) ────────────────────────────


@case(tags=["symbol"])
def case_sql_alter_table() -> SymbolCase:
    """ALTER TABLE → function, named by the table."""
    return "sql", "ALTER TABLE users ADD COLUMN age INT;\n", [("function", "users", "")]


@case(tags=["symbol"])
def case_sql_drop_table() -> SymbolCase:
    """DROP TABLE → function, named by the table."""
    return "sql", "DROP TABLE legacy;\n", [("function", "legacy", "")]


@case(tags=["symbol"])
def case_sql_drop_index() -> SymbolCase:
    """DROP INDEX → function, named by the index."""
    return "sql", "DROP INDEX idx_old;\n", [("function", "idx_old", "")]


# ── Naming invariances (optional clauses must not change the name) ───


@case(tags=["symbol"])
def case_sql_table_if_not_exists() -> SymbolCase:
    """IF NOT EXISTS does not change the table name."""
    return "sql", "CREATE TABLE IF NOT EXISTS users (id INT);\n", [("class", "users", "")]


@case(tags=["symbol"])
def case_sql_view_or_replace() -> SymbolCase:
    """CREATE OR REPLACE does not change the view name."""
    return "sql", "CREATE OR REPLACE VIEW v AS SELECT 1;\n", [("class", "v", "")]


@case(tags=["symbol"])
def case_sql_schema_qualified_table() -> SymbolCase:
    """A schema-qualified table is named by the table, not the schema."""
    return "sql", "CREATE TABLE app.users (id INT);\n", [("class", "users", "")]


@case(tags=["symbol"])
def case_sql_select_aliased_table() -> SymbolCase:
    """An aliased FROM is named by the table, not the alias."""
    src = "SELECT c.id FROM chunks AS c JOIN files f ON f.id = c.id;\n"
    return "sql", src, [("function", "chunks", "")]


@case(tags=["symbol"])
def case_sql_create_table_as_select() -> SymbolCase:
    """CREATE TABLE AS SELECT yields only the table, not the nested select."""
    return "sql", "CREATE TABLE snap AS SELECT * FROM users;\n", [("class", "snap", "")]


@case(tags=["symbol"])
def case_sql_insert_select() -> SymbolCase:
    """INSERT ... SELECT is named by the insert target."""
    src = "INSERT INTO archive SELECT * FROM events;\n"
    return "sql", src, [("function", "archive", "")]


# ── Anonymous statements (no nameable target) ────────────────────────


@case(tags=["symbol"])
def case_sql_select_no_table() -> SymbolCase:
    """A SELECT with no table is anonymous."""
    return "sql", "SELECT 1;\n", [("function", "<anonymous>", "")]


@case(tags=["symbol"])
def case_sql_union() -> SymbolCase:
    """A top-level UNION (set_operation) is one anonymous function chunk."""
    src = "SELECT id FROM a UNION SELECT id FROM b;\n"
    return "sql", src, [("function", "<anonymous>", "")]


@case(tags=["symbol"])
def case_sql_param_error_recovery() -> SymbolCase:
    """A DuckDB $param errors internally but the statement still extracts.

    The `$id` placeholder is unknown to the grammar (an ERROR node),
    but error recovery keeps the enclosing statement, so the chunk is
    still produced and named by its table.
    """
    return "sql", "SELECT * FROM users WHERE id = $id;\n", [("function", "users", "")]


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
def case_sql_migration() -> MixedCase:
    """Realistic migration script producing every SQL chunk kind.

    SQL has no method scoping, so the expected-methods list is
    empty — definitions are all top-level.
    """
    src = """\
CREATE SCHEMA shop;

CREATE TABLE shop.products (
    id INT PRIMARY KEY,
    name TEXT NOT NULL,
    price NUMERIC
);

CREATE VIEW in_stock AS
SELECT * FROM shop.products WHERE price > 0;

CREATE FUNCTION price_with_tax(p NUMERIC) RETURNS NUMERIC
LANGUAGE SQL
AS $$ SELECT p * 1.2; $$;

CREATE INDEX idx_products_name ON shop.products (name);
"""
    return (
        "sql",
        src,
        {"class", "function", "variable"},
        [],
    )
