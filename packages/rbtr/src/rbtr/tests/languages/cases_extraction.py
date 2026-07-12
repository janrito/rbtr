"""Extraction test cases for all languages.

Each `@case` function returns a tuple of test data consumed by
`test_symbol_extraction.py` and `test_import_extraction.py`
via `@parametrize_with_cases`.

Organisation: one section per language, cases tagged by behavior
(`symbol`, `import`, `multi_import`, `mixed`).
"""

from __future__ import annotations

import pytest
from pytest_cases import case

# Return types per tag — each @case function returns one of these.
type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str]]
type MultiImportCase = tuple[str, str, int, list[dict[str, str]]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]


# ═════════════════════════════════════════════════════════════════════
# Python
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


@case(tags=["symbol"])
def case_py_simple_function() -> SymbolCase:
    """Top-level function."""
    src = """\
def hello():
    pass
"""
    return "python", src, [("function", "hello", "")]


@case(tags=["symbol"])
def case_py_function_with_args() -> SymbolCase:
    """Function with parameters."""
    src = """\
def add(a, b):
    return a + b
"""
    return "python", src, [("function", "add", "")]


@case(tags=["symbol"])
def case_py_async_function() -> SymbolCase:
    """Async function."""
    src = """\
async def fetch():
    pass
"""
    return "python", src, [("function", "fetch", "")]


@case(tags=["symbol"])
def case_py_multiple_functions() -> SymbolCase:
    """Multiple functions — all present."""
    src = """\
def foo():
    pass

def bar():
    pass

def baz():
    pass
"""
    return (
        "python",
        src,
        [
            ("function", "foo", ""),
            ("function", "bar", ""),
            ("function", "baz", ""),
        ],
    )


@case(tags=["symbol"])
def case_py_decorated_function() -> SymbolCase:
    """Decorated function."""
    src = """\
@decorator
def wrapped():
    pass
"""
    return "python", src, [("function", "wrapped", "")]


@case(tags=["symbol"])
def case_py_simple_class() -> SymbolCase:
    """Top-level class."""
    src = """\
class Foo:
    pass
"""
    return "python", src, [("class", "Foo", "")]


@case(tags=["symbol"])
def case_py_class_with_bases() -> SymbolCase:
    """Class with inheritance."""
    src = """\
class Bar(Foo, Mixin):
    pass
"""
    return "python", src, [("class", "Bar", "")]


@case(tags=["symbol"])
def case_py_decorated_class() -> SymbolCase:
    """Decorated class."""
    src = """\
@dataclass
class Config:
    name: str
"""
    return "python", src, [("class", "Config", "")]


@case(tags=["symbol"])
def case_py_multiple_classes() -> SymbolCase:
    """Multiple classes."""
    src = """\
class A:
    pass

class B:
    pass
"""
    return "python", src, [("class", "A", ""), ("class", "B", "")]


@case(tags=["symbol"])
def case_py_method_in_class() -> SymbolCase:
    """Method scoped to class."""
    src = """\
class Foo:
    def bar(self):
        pass
"""
    return "python", src, [("method", "bar", "Foo")]


@case(tags=["symbol"])
def case_py_multiple_methods() -> SymbolCase:
    """Multiple methods in one class."""
    src = """\
class Svc:
    def start(self):
        pass
    def stop(self):
        pass
"""
    return "python", src, [("method", "start", "Svc"), ("method", "stop", "Svc")]


@case(tags=["symbol"])
def case_py_nested_class_method() -> SymbolCase:
    """Method in a nested class carries the full class path."""
    src = """\
class Outer:
    class Inner:
        def deep(self):
            pass
"""
    return "python", src, [("method", "deep", "Outer::Inner")]


@case(tags=["symbol"])
def case_py_top_level_not_method() -> SymbolCase:
    """Function after a class is not scoped."""
    src = """\
class Foo:
    pass

def standalone():
    pass
"""
    return "python", src, [("function", "standalone", "")]


@case(tags=["symbol"])
def case_py_static_method_scoped() -> SymbolCase:
    """Staticmethod still scoped to class."""
    src = """\
class Svc:
    @staticmethod
    def create():
        pass
"""
    return "python", src, [("method", "create", "Svc")]


@case(tags=["symbol"])
def case_py_closure_in_function() -> SymbolCase:
    """A closure is addressed by its enclosing function and stays a function.

    Function nesting must contribute to the address (inner `handler`
    → scope "make_adder"), and a function nested in a function must
    remain kind=function — it is not a method.
    """
    src = """\
def make_adder(n):
    def handler():
        return n + 1
    return handler
"""
    return "python", src, [("function", "handler", "make_adder")]


@case(tags=["symbol"])
def case_py_closure_in_method_stays_function() -> SymbolCase:
    """A function nested in a method is addressed fully and NOT promoted.

    The enclosing scope path is "Svc::start" and `cb` must stay
    kind=function — promotion follows the nearest *enclosing scope
    node's* type (a class), not merely a non-empty scope.
    """
    src = """\
class Svc:
    def start(self):
        def cb():
            return 1
        return cb
"""
    return "python", src, [("function", "cb", "Svc::start")]


@case(tags=["symbol"])
def case_py_repeated_nested_name() -> SymbolCase:
    """Repeated scope names compose without collapsing.

    Three classes all named `Node` nest; the innermost method's path
    keeps every level (`Node::Node::Node`), not a deduplicated one.
    """
    src = """\
class Node:
    class Node:
        class Node:
            def visit(self):
                pass
"""
    return "python", src, [("method", "visit", "Node::Node::Node")]


@case(tags=["symbol"])
def case_py_shadowed_closure() -> SymbolCase:
    """A closure shadowing its enclosing function's name stays distinct.

    The inner `process` is addressed `process` (its outer namesake);
    the two share a name but differ in scope, so addressing keeps
    them apart. The inner one is a function, not a method.
    """
    src = """\
def process():
    def process():
        return 1
    return process
"""
    return "python", src, [("function", "process", "process"), ("function", "process", "")]


@case(tags=["symbol"])
def case_py_mixed_deep_nesting() -> SymbolCase:
    """Mixed class/method/closure/class chain composes the full path.

    `deep` sits in a local class, in a closure, in a method, in a
    class — so its path mixes every scope kind. Promotion follows the
    *nearest* scope: `helper` (in a method) is a function, `deep` (in
    a class) is a method.
    """
    src = """\
class Outer:
    def run(self):
        def helper():
            class Local:
                def deep(self):
                    pass
            return Local
        return helper
"""
    return (
        "python",
        src,
        [
            ("method", "run", "Outer"),
            ("function", "helper", "Outer::run"),
            ("class", "Local", "Outer::run::helper"),
            ("method", "deep", "Outer::run::helper::Local"),
        ],
    )


@case(tags=["symbol"])
def case_py_method_named_like_class() -> SymbolCase:
    """A method whose name equals its class is addressed `Node::`, not merged.

    The method `Node` inside class `Node` is `(method, Node, Node)` —
    name and scope segment coincide but are different objects; the
    address keeps both.
    """
    src = """\
class Node:
    def Node(self):
        return 1
"""
    return "python", src, [("class", "Node", ""), ("method", "Node", "Node")]


@case(tags=["symbol"])
def case_py_method_named_like_nested_class() -> SymbolCase:
    """Name equal to a *repeated* scope segment still composes fully.

    A class `Node` in a class `Node` with a method `Node`: the method
    is `(method, Node, Node::Node)` and the inner class is
    `(class, Node, Node)`.
    """
    src = """\
class Node:
    class Node:
        def Node(self):
            return 1
"""
    return "python", src, [("class", "Node", "Node"), ("method", "Node", "Node::Node")]


@case(tags=["symbol"])
def case_py_function_and_class_same_name() -> SymbolCase:
    """A function and a class sharing a name at module scope both extract.

    Different objects (`function` and `class`) collide on identity
    `(name, scope)` = `(Cache, "")` — the residual same-scope
    collision addressing cannot split (kind is not part of identity).
    Both must appear; the diff's content-set backstop keeps them apart.
    """
    src = """\
def Cache():
    return None

class Cache:
    pass
"""
    return "python", src, [("function", "Cache", ""), ("class", "Cache", "")]


@case(tags=["symbol"])
def case_py_module_constant() -> SymbolCase:
    """Module-level constant."""
    src = """\
MAX_SIZE = 100
"""
    return "python", src, [("variable", "MAX_SIZE", "")]


@case(tags=["symbol"])
def case_py_module_singleton() -> SymbolCase:
    """Module-level lowercase singleton (the motivating case)."""
    src = """\
config = Config()
"""
    return "python", src, [("variable", "config", "")]


@case(tags=["symbol"])
def case_py_annotated_module_var() -> SymbolCase:
    """Module-level annotated assignment."""
    src = """\
TIMEOUT: int = 30
"""
    return "python", src, [("variable", "TIMEOUT", "")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
def case_py_import_bare() -> ImportCase:
    """import os."""
    return "python", "import os\n", {"module": "os"}


@case(tags=["import"])
def case_py_import_dotted() -> ImportCase:
    """import os.path."""
    return "python", "import os.path\n", {"module": "os.path"}


@case(tags=["import"])
def case_py_import_deeply_nested() -> ImportCase:
    """import a.b.c.d."""
    return "python", "import a.b.c.d\n", {"module": "a.b.c.d"}


@case(tags=["import"])
def case_py_from_import_single() -> ImportCase:
    """from pathlib import Path."""
    return "python", "from pathlib import Path\n", {"module": "pathlib", "names": "Path"}


@case(tags=["import"])
def case_py_from_import_multiple() -> ImportCase:
    """from module import multiple names."""
    return (
        "python",
        "from rbtr.index.models import Chunk, Edge\n",
        {"module": "rbtr.index.models", "names": "Chunk,Edge"},
    )


@case(tags=["import"])
def case_py_from_import_aliased() -> ImportCase:
    """Aliased import extracts original name."""
    return (
        "python",
        "from .models import Chunk as C\n",
        {"dots": "1", "module": "models", "names": "Chunk"},
    )


@case(tags=["import"])
def case_py_from_import_multiple_aliased() -> ImportCase:
    """Multiple aliased imports."""
    return (
        "python",
        "from models import Foo as F, Bar as B\n",
        {"module": "models", "names": "Foo,Bar"},
    )


@case(tags=["import"])
def case_py_relative_dot_with_module() -> ImportCase:
    """from .models import Chunk."""
    return (
        "python",
        "from .models import Chunk\n",
        {"dots": "1", "module": "models", "names": "Chunk"},
    )


@case(tags=["import"])
def case_py_relative_dotdot() -> ImportCase:
    """from ..core import engine."""
    return (
        "python",
        "from ..core import engine\n",
        {"dots": "2", "module": "core", "names": "engine"},
    )


@case(tags=["import"])
def case_py_relative_dot_only() -> ImportCase:
    """from . import utils — no module key."""
    return "python", "from . import utils\n", {"dots": "1", "names": "utils"}


@case(tags=["import"])
def case_py_relative_three_dots() -> ImportCase:
    """from ...lib import helper."""
    return (
        "python",
        "from ...lib import helper\n",
        {"dots": "3", "module": "lib", "names": "helper"},
    )


@case(tags=["import"])
def case_py_import_star() -> ImportCase:
    """from os.path import * — no names key."""
    return "python", "from os.path import *\n", {"module": "os.path"}


@case(tags=["import"])
def case_py_import_inside_function() -> ImportCase:
    """Nested import still captured."""
    src = """\
def f():
    import json
"""
    return "python", src, {"module": "json"}


@case(tags=["import"])
def case_py_import_inside_class() -> ImportCase:
    """Import inside class body."""
    src = """\
class C:
    from collections import OrderedDict
"""
    return "python", src, {"module": "collections", "names": "OrderedDict"}


# ── Multi-import ─────────────────────────────────────────────────────


@case(tags=["multi_import"])
def case_py_multiple_imports() -> MultiImportCase:
    """Two bare import statements."""
    src = """\
import os
import sys
"""
    return "python", src, 2, [{"module": "os"}, {"module": "sys"}]


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
def case_py_full_module() -> MixedCase:
    """Realistic module with all symbol types and docstrings.

    Docstrings are PEP-257 style on every symbol.  The
    expected-tuple pins symbol extraction invariants; content
    invariants are covered separately by `test_docstrings.py`.
    Adding docs here exercises the realistic shape that
    production Python code has.
    """
    src = '''\
"""Module-level docstring for the Config helper."""

import os
from pathlib import Path


class Config:
    """Runtime configuration."""

    def __init__(self):
        """Initialise with defaults."""
        pass

    def load(self):
        """Load from disk."""
        pass


def main():
    """Entry point."""
    pass
'''
    return (
        "python",
        src,
        {"import", "class", "method", "function"},
        [("__init__", "Config"), ("load", "Config")],
    )


# ═════════════════════════════════════════════════════════════════════
# JavaScript
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
def case_md_splits_by_heading() -> SymbolCase:
    """Markdown heading hierarchy."""
    src = """\
# Title

Intro text.

## Section A

Body A.

## Section B

Body B.
"""
    return (
        "markdown",
        src,
        [
            ("doc_section", "Title", ""),
            ("doc_section", "Section A", "Title"),
            ("doc_section", "Section B", "Title"),
        ],
    )


@case(tags=["symbol"])
def case_md_scope_chain() -> SymbolCase:
    """Nested heading scope."""
    src = """\
# Top

## Mid

### Deep

Content here.
"""
    return "markdown", src, [("doc_section", "Deep", "Top::Mid")]


@case(tags=["symbol"])
def case_md_same_name_under_different_parents() -> SymbolCase:
    """Same-named sections under different parents get distinct scopes.

    Two `Overview` subsections — one under `A`, one under `B` — are
    `(doc_section, Overview, A)` and `(doc_section, Overview, B)`. Without
    the full heading path they would collide on identity; addressing
    keeps them apart.
    """
    src = """\
# A

Alpha intro.

## Overview

Alpha overview.

# B

Beta intro.

## Overview

Beta overview.
"""
    return (
        "markdown",
        src,
        [("doc_section", "Overview", "A"), ("doc_section", "Overview", "B")],
    )


# ══════════════════════════════��══════════════════════════════════════
# RST
# ═════���════════════��═══════════════════════════════════��══════════════


@case(tags=["symbol"])
def case_rst_splits_by_heading() -> SymbolCase:
    """RST heading hierarchy."""
    src = """\
Title
=====

Intro text.

Section A
---------

Body A.

Section B
---------

Body B.
"""
    return (
        "rst",
        src,
        [
            ("doc_section", "Title", ""),
            ("doc_section", "Section B", "Title"),
        ],
    )


@case(tags=["symbol"])
def case_rst_scope_chain() -> SymbolCase:
    """RST nested scope from adornment order."""
    src = """\
Top
===

Mid
---

Deep
^^^

Content here.
"""
    return "rst", src, [("doc_section", "Deep", "Top::Mid")]


@case(tags=["symbol"])
def case_rst_same_name_under_different_parents() -> SymbolCase:
    """Same-named RST subsections under different parents stay distinct.

    Both `Overview` subsections are scoped by their parent only — `A`
    and `B` — never themselves, so they stay distinct.
    """
    src = """\
A
=

Alpha intro.

Overview
--------

Alpha overview.

B
=

Beta intro.

Overview
--------

Beta overview.
"""
    return (
        "rst",
        src,
        [("doc_section", "Overview", "A"), ("doc_section", "Overview", "B")],
    )


# ═════════════════════��════════════════════════════════════��══════════
# JSON
# ════════════════════════════════════════════════���════════════════════


@case(tags=["symbol"])
def case_json_top_level_keys() -> SymbolCase:
    """JSON splits by top-level keys."""
    src = """\
{
  "name": "my-project",
  "version": "1.0.0",
  "dependencies": {
    "foo": "^1.0"
  }
}
"""
    return (
        "json",
        src,
        [
            ("doc_section", "name", ""),
            ("doc_section", "version", ""),
            ("doc_section", "dependencies", ""),
        ],
    )


# ══════════════════════���══════════════════════════════════════════════
# CSS
# ════════════════���══════════════════════════��═════════════════════════


@case(tags=["symbol"])
def case_css_rule_sets() -> SymbolCase:
    """CSS splits by rule sets."""
    src = """\
body {
  color: #333;
}

.header {
  background: blue;
}
"""
    return (
        "css",
        src,
        [
            ("doc_section", "body", ""),
            ("doc_section", ".header", ""),
        ],
    )


# ══════════════════════════════════════════════════��══════════════════
# TOML
# ════════════��════════════════════════════════════════════════════════


@case(tags=["symbol"])
def case_toml_splits_by_table() -> SymbolCase:
    """TOML splits by tables; a dotted table is named by its last segment."""
    src = """\
[project]
name = "rbtr"

[tool.ruff]
line-length = 99
"""
    return (
        "toml",
        src,
        [
            ("doc_section", "project", ""),
            ("doc_section", "ruff", "tool"),
        ],
    )


# ═��═════════════════════════════════════════��═════════════════════════
# YAML
# ═══════════════════════════════════════════════════════════��═════════


@case(tags=["symbol"])
def case_yaml_top_level_keys() -> SymbolCase:
    """YAML splits by top-level mapping keys."""
    src = """\
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
"""
    return (
        "yaml",
        src,
        [
            ("doc_section", "name", ""),
            ("doc_section", "on", ""),
            ("doc_section", "jobs", ""),
        ],
    )


# ═════���══════���════════════════════════════════════════════════════════
# HCL
# ═══��═══════════════════════════════════════���═════════════════════════


@case(tags=["symbol"])
def case_hcl_splits_by_blocks() -> SymbolCase:
    """HCL splits by top-level blocks."""
    src = """\
resource "aws_instance" "web" {
  ami = "ami-12345"
}

variable "region" {
  default = "us-east-1"
}
"""
    return (
        "hcl",
        src,
        [
            ("doc_section", "resource aws_instance web", ""),
            ("doc_section", "variable region", ""),
        ],
    )


# ═════════════���═══════════════════════════════════════════════════════
# HTML
# ════════════════════════════════════════════════════════════════════��


@case(tags=["symbol"])
def case_html_semantic_elements() -> SymbolCase:
    """HTML names an element by its id (even beside a class), else its tag."""
    src = """\
<html>
<body>
  <main id="content"><p>Hello</p></main>
  <article class="post" id="first">Text</article>
  <nav>Links</nav>
</body>
</html>
"""
    return (
        "html",
        src,
        [
            ("doc_section", "body", ""),
            ("doc_section", "content", ""),
            ("doc_section", "first", ""),
            ("doc_section", "nav", ""),
        ],
    )


# ═════════════════════════════════════════════════════════════════════
# SQL
# ═════════════════════════════════════════════════════════════════════
#
# The cases below are the spec of the SQL constructs we support. One
# chunk is produced per top-level statement (plus one per CTE), named by
# its object/table; SQL has no nesting so every scope is "". Construct
# cases pin (kind, name); invariance cases prove optional clauses don't
# change the name; edge cases cover anonymous statements, error recovery,
# and the known grammar gaps. SQL keywords are uppercase by convention.

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


# ═════════════════════════════════════════════════════════════════════
# Module-level variables (cross-language fan-out)
# ═════════════════════════════════════════════════════════════════════


@case(tags=["symbol"])
def case_py_tuple_unpack() -> SymbolCase:
    """Flat tuple unpacking."""
    return "python", "a, b = compute()\n", [("variable", "a", ""), ("variable", "b", "")]


@case(tags=["symbol"])
def case_py_paren_tuple_unpack() -> SymbolCase:
    """Parenthesised tuple target."""
    return "python", "(a, b) = compute()\n", [("variable", "a", ""), ("variable", "b", "")]


@case(tags=["symbol"])
def case_py_list_unpack() -> SymbolCase:
    """List-pattern target."""
    return "python", "[a, b] = compute()\n", [("variable", "a", ""), ("variable", "b", "")]


@case(tags=["symbol"])
def case_py_star_unpack() -> SymbolCase:
    """Starred target."""
    return "python", "a, *rest = compute()\n", [("variable", "a", ""), ("variable", "rest", "")]


@case(tags=["symbol"], marks=_xfail_nested)
def case_py_nested_unpack_xfail() -> SymbolCase:
    """Nested tuple unpacking — only the outer level is captured today."""
    return (
        "python",
        "(a, b), c = f()\n",
        [("variable", "a", ""), ("variable", "b", ""), ("variable", "c", "")],
    )


@case(tags=["symbol"], marks=_xfail_nested)
def case_py_chained_assignment_xfail() -> SymbolCase:
    """Chained assignment — only the first target is captured today."""
    return "python", "a = b = f()\n", [("variable", "a", ""), ("variable", "b", "")]
