"""SQL language plugin.

Provides symbol extraction for DDL definitions. SQL has no
functions/classes in the OO sense and no native import mechanism,
so constructs map onto rbtr's capture conventions by shape:
routines are functions, structural definitions are classes, and
standalone named objects are variables.

Extracted chunks::

    CREATE FUNCTION add(...) ...        → function "add", scope ""

    CREATE TABLE users (...)            → class "users", scope ""
    CREATE VIEW active AS ...           → class "active", scope ""
    CREATE MATERIALIZED VIEW mv AS ...  → class "mv", scope ""
    CREATE TYPE mood AS ENUM (...)      → class "mood", scope ""

    CREATE INDEX idx ON users(...)      → variable "idx", scope ""
    CREATE SEQUENCE seq ...             → variable "seq", scope ""
    CREATE SCHEMA app                   → variable "app", scope ""
    CREATE DATABASE db                  → variable "db", scope ""
    CREATE EXTENSION postgis            → variable "postgis", scope ""
    CREATE ROLE admin                   → variable "admin", scope ""
    CREATE TRIGGER trg BEFORE ... ON t  → variable "trg", scope ""

`CREATE PROCEDURE` is not extracted: the tree-sitter-sql grammar
(0.3.11) has no `create_procedure` node — it produces an `ERROR`
node and is skipped.
"""

from __future__ import annotations

from rbtr.languages.hookspec import LanguageRegistration, hookimpl

# ── Query ────────────────────────────────────────────────────────────

# Names sit on a nested `object_reference` for most definitions;
# index/schema/database/extension/role names are a bare `identifier`
# child. The trigger name is the `object_reference` anchored
# immediately after `keyword_trigger`, distinguishing it from the
# ON-table and EXECUTE-FUNCTION references in the same statement.
_QUERY = """\
(create_function
  (object_reference name: (identifier) @_fn_name)) @function

(create_table
  (object_reference name: (identifier) @_cls_name)) @class

(create_view
  (object_reference name: (identifier) @_cls_name)) @class

(create_materialized_view
  (object_reference name: (identifier) @_cls_name)) @class

(create_type
  (object_reference name: (identifier) @_cls_name)) @class

(create_index
  (identifier) @_var_name) @variable

(create_sequence
  (object_reference name: (identifier) @_var_name)) @variable

(create_schema
  (identifier) @_var_name) @variable

(create_database
  (identifier) @_var_name) @variable

(create_extension
  (identifier) @_var_name) @variable

(create_role
  (identifier) @_var_name) @variable

(create_trigger
  (keyword_trigger) . (object_reference name: (identifier) @_var_name)) @variable
"""

# ── Plugin ───────────────────────────────────────────────────────────


class SqlPlugin:
    """SQL language support — DDL definitions, no imports."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="sql",
                extensions=frozenset({".sql"}),
                grammar_module="tree_sitter_sql",
                query=_QUERY,
                # SQL `--` line comments and `/* */` blocks both
                # parse to `comment`; attach a leading run to its
                # following definition, matching the Go/Ruby
                # convention.
                doc_comment_node_types=frozenset({"comment"}),
                language_plugin_version=1,
            ),
        ]
