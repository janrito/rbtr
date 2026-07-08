"""Cases for per-language sample extraction tests.

Two tag-separated families share this file:

- `sample` — one case per language, returning `(lang, expected_kinds)`.
  The fixture loads the committed `samples/<lang>/` mini-project (one or
  more files) from `samples/<lang>/` and feeds both the chunk and edge
  snapshots.
- `unsupported` — constructs that *should* extract but do not yet;
  each is `xfail(strict=True)` and returns `(lang, snippet,
  expected_symbol)`. Populated from Phase 2 onward.

SQL sample and dialect cases moved to the `rbtr-lang-sql` package.
"""

from __future__ import annotations

import pytest
from pytest_cases import case

from rbtr.index.models import ChunkKind

type SampleCase = tuple[str, set[ChunkKind]]
type UnsupportedCase = tuple[str, str, tuple[ChunkKind, str, str]]


@case(id="python", tags=["sample"])
def case_python() -> SampleCase:
    """Python: sync/async/decorated functions, classes, methods
    (instance/property/static/class), module variables incl. tuple
    unpacking, and aliased/relative imports with docstrings.
    """
    return (
        "python",
        {
            ChunkKind.FUNCTION,
            ChunkKind.CLASS,
            ChunkKind.METHOD,
            ChunkKind.VARIABLE,
            ChunkKind.IMPORT,
        },
    )


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


@case(id="html", tags=["sample"])
def case_html() -> SampleCase:
    """HTML: <script src> and <link href> as imports (with a language
    hint, plain and self-closing tags), and major <body> elements as doc
    sections. Inline <script>/<style> delegate to JavaScript (a function)
    and CSS (a rule set).
    """
    return (
        "html",
        {ChunkKind.IMPORT, ChunkKind.DOC_SECTION, ChunkKind.FUNCTION},
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


@case(id="query", tags=["sample"])
def case_query() -> SampleCase:
    """tree-sitter query (`.scm`): every top-level pattern is a doc section,
    named by its definition capture (else its matched node type). Exercises
    named nodes, alternation lists, groupings, predicates, wildcards, negated
    fields, anchors, quantifiers, supertypes, and a MISSING node.
    """
    return (
        "query",
        {ChunkKind.DOC_SECTION},
    )


@case(id="rst", tags=["sample"])
def case_rst() -> SampleCase:
    """reStructuredText: heading hierarchy from adornment order (as doc
    sections), and :doc:/:func:/:class:/:mod: roles, references, and
    toctree entries (as imports). External references are skipped.
    """
    return (
        "rst",
        {ChunkKind.DOC_SECTION, ChunkKind.IMPORT},
    )


@case(id="markdown", tags=["sample"])
def case_markdown() -> SampleCase:
    """Markdown: heading hierarchy (as doc sections with `::` scope) and
    local links (as imports, with `names` from a `#fragment`). External
    and fragment-only links are skipped. Fenced code blocks delegate to
    their language: a ```python``` block yields an import and a function,
    a ```sh``` block a function — proving both hint-resolution paths.
    """
    return (
        "markdown",
        {ChunkKind.DOC_SECTION, ChunkKind.IMPORT, ChunkKind.FUNCTION},
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


@case(id="svelte", tags=["sample"])
def case_svelte() -> SampleCase:
    """Svelte SFC: the `<script lang="ts">` block is delegated to TypeScript
    (functions, exported variables, imports) and the `<style lang="scss">`
    block to SCSS ($-variables, rule sets as doc sections). The markup
    template is extracted as one host `svelte` doc-section chunk.
    """
    return (
        "svelte",
        {
            ChunkKind.FUNCTION,
            ChunkKind.VARIABLE,
            ChunkKind.DOC_SECTION,
            ChunkKind.IMPORT,
        },
    )


@case(id="vue", tags=["sample"])
def case_vue() -> SampleCase:
    """Vue SFC: the `<script setup lang="ts">` block is delegated to
    TypeScript (functions, consts as variables, imports) and the
    `<style lang="scss">` block to SCSS ($-variables, rule sets as doc
    sections), reusing the Svelte chunker. The `<template>` is extracted as
    one host `vue` doc-section chunk.
    """
    return (
        "vue",
        {
            ChunkKind.FUNCTION,
            ChunkKind.VARIABLE,
            ChunkKind.DOC_SECTION,
            ChunkKind.IMPORT,
        },
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


@case(id="ruby", tags=["sample"])
def case_ruby() -> SampleCase:
    """Ruby: top-level defs (as functions), classes and modules (both as
    classes), defs inside them (as methods, incl. `self.` singletons),
    constant assignments (as variables), and require/require_relative.
    """
    return (
        "ruby",
        {
            ChunkKind.FUNCTION,
            ChunkKind.CLASS,
            ChunkKind.METHOD,
            ChunkKind.VARIABLE,
            ChunkKind.IMPORT,
        },
    )


# ── Known-unsupported constructs (xfail registry) ────────────────────
#
# Each asserts the *ideal* chunk a construct should produce; all are
# xfail(strict) so the suite fails loudly if a plugin upgrade closes the
# gap. See plan Appendix E.


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
