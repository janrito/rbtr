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

from pytest_cases import case

from rbtr.index.models import ChunkKind

type SampleCase = tuple[str, set[ChunkKind]]
type UnsupportedCase = tuple[str, str, tuple[ChunkKind, str, str]]
type SqlDialectCase = tuple[str, str]


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
