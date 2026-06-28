"""SQL language plugin.

SQL has no functions/classes in the OO sense and no native import
mechanism, so constructs map onto rbtr's capture conventions by shape:
structural definitions are classes, routines and executable statements
are functions, and standalone named objects are variables.  One chunk is
produced per top-level statement, plus one per CTE.

The tree-sitter query is *generated* from `_DDL_VERBS` (one pattern per
DDL/ALTER/DROP verb) and combined with hand-written patterns for the DML
statements whose structure the table can't express.  The verb set and
each verb's name location are verified against `tree-sitter-sql` 0.3.11.

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

from rbtr.languages.hookspec import LanguageRegistration, hookimpl

# ── Query generation ─────────────────────────────────────────────────

# Capture name carrying the symbol name, per chunk kind.
_CAPTURE = {"class": "_cls_name", "function": "_fn_name", "variable": "_var_name"}

# DDL/ALTER/DROP verbs: (node_type, chunk_kind, name_node).
#   name_node "object_reference" — the verb's single target object_reference
#     (table/view/type/sequence/function).  Unanchored: each of these verbs
#     has exactly one direct object_reference, and leaving it unanchored
#     keeps `IF NOT EXISTS` / `OR REPLACE` / `TEMP` working (an anchor to
#     the type keyword breaks when those keywords intervene).
#   name_node "identifier" — a bare identifier child (index/schema/role/…);
#     for these the object_reference, if any, is a *secondary* reference
#     (e.g. the `ON` table of an index) so the bare identifier is the name.
# Verified against tree-sitter-sql 0.3.11.  `create_trigger` is handled
# separately (three object_references — the name must be anchored).
_DDL_VERBS: list[tuple[str, str, str]] = [
    ("create_table", "class", "object_reference"),
    ("create_view", "class", "object_reference"),
    ("create_materialized_view", "class", "object_reference"),
    ("create_type", "class", "object_reference"),
    ("create_function", "function", "object_reference"),
    ("create_sequence", "variable", "object_reference"),
    ("create_index", "variable", "identifier"),
    ("create_schema", "variable", "identifier"),
    ("create_database", "variable", "identifier"),
    ("create_extension", "variable", "identifier"),
    ("create_role", "variable", "identifier"),
    ("alter_table", "function", "object_reference"),
    ("alter_view", "function", "object_reference"),
    ("alter_sequence", "function", "object_reference"),
    ("alter_index", "function", "identifier"),
    ("alter_schema", "function", "identifier"),
    ("alter_database", "function", "identifier"),
    ("alter_role", "function", "identifier"),
    ("alter_type", "function", "identifier"),
    ("drop_table", "function", "object_reference"),
    ("drop_view", "function", "object_reference"),
    ("drop_sequence", "function", "object_reference"),
    ("drop_function", "function", "object_reference"),
    ("drop_type", "function", "object_reference"),
    ("drop_index", "function", "identifier"),
    ("drop_schema", "function", "identifier"),
    ("drop_database", "function", "identifier"),
    ("drop_role", "function", "identifier"),
    ("drop_extension", "function", "identifier"),
]


def _ddl_pattern(node_type: str, kind: str, name_node: str) -> str:
    """Build a query pattern for a DDL/ALTER/DROP verb node."""
    cap = _CAPTURE[kind]
    if name_node == "object_reference":
        inner = f"(object_reference name: (identifier) @{cap})"
    else:
        inner = f"(identifier) @{cap}"
    return f"({node_type} {inner}) @{kind}"


# DML statements, CTEs, triggers, and set operations — structures the DDL
# table can't express.  DML patterns are anchored at `(program (statement
# …))` so each top-level statement yields exactly one chunk: `select` and
# `delete` keep their target in a sibling `from`, so the whole `statement`
# is captured.  A `set_operation` (UNION/INTERSECT/EXCEPT) has no single
# table and is named `<anonymous>`.
_OTHER_PATTERNS = """\
(create_trigger (keyword_trigger) . (object_reference name: (identifier) @_var_name)) @variable

(cte (identifier) @_fn_name) @function

(program (statement (select)
  (from (relation (object_reference name: (identifier) @_fn_name)))?) @function)
(program (statement (set_operation) @function))
(program (statement (insert (object_reference name: (identifier) @_fn_name))) @function)
(program (statement (update (relation (object_reference name: (identifier) @_fn_name)))) @function)
(program (statement (delete) (from (object_reference name: (identifier) @_fn_name))) @function)
"""

_QUERY = "\n".join(_ddl_pattern(*verb) for verb in _DDL_VERBS) + "\n\n" + _OTHER_PATTERNS

# ── Plugin ───────────────────────────────────────────────────────────


class SqlPlugin:
    """SQL language support — DDL definitions, DML statements, and CTEs."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="sql",
                extensions=frozenset({".sql"}),
                grammar_module="tree_sitter_sql",
                query=_QUERY,
                # SQL `--` line comments and `/* */` blocks both parse to
                # `comment`; attach a leading run to its statement, as the
                # Go/Ruby plugins do.
                doc_comment_node_types=frozenset({"comment"}),
                language_plugin_version=2,
            ),
        ]
