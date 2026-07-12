"""SQL language plugin.

SQL has no functions/classes in the OO sense and no native import
mechanism, so constructs map onto rbtr's capture conventions by shape:
structural definitions are classes, routines and executable statements
are functions, and standalone named objects are variables.  One chunk is
produced per top-level statement, plus one per CTE.

The tree-sitter query lives in `sql.scm`: DDL/ALTER/DROP verbs grouped
into `[...]` alternations by chunk kind and name location, plus hand-
written patterns for the DML statements those groups can't express.
Verified against `tree-sitter-sql` 0.3.11.

Extracted chunks::

    CREATE TABLE users (...)            → class "users"
    CREATE VIEW / MATERIALIZED VIEW     → class
    CREATE TYPE mood AS ENUM (...)      → class "mood"

    CREATE FUNCTION add(...) ...        → function "add"
    SELECT ... FROM users               → function "users"
    INSERT INTO logs ...                → function "logs"
    UPDATE / DELETE ... t               → function "t"
    WITH ranked AS (...)                → function "ranked"  (one per CTE)
    ... UNION ...                       → function "<anonymous>"

    CREATE INDEX idx ON users(...)      → variable "idx"
    CREATE SEQUENCE / SCHEMA / ROLE …   → variable
    CREATE TRIGGER trg ...              → variable "trg"
    ALTER / DROP <object>               → function "<object>"

A statement with no nameable target (`SELECT 1`, a sub-query `FROM`, a
`UNION`) is named `<anonymous>`.  `CREATE PROCEDURE` and `PRAGMA` are not
extracted — the grammar has no node for them (they parse to `ERROR`).
"""

from __future__ import annotations

from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

# ── Plugin ───────────────────────────────────────────────────────────


sql = LanguageRegistration(
    id="sql",
    extensions=frozenset({".sql"}),
    grammar_module="tree_sitter_sql",
    extraction=QueryExtraction(
        query=load_query(__package__, "sql"),
    ),
    extraction_serial=3,
)
