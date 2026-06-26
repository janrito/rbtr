"""Tests for language extraction — symbols, imports, and properties.

All test data lives in `case_extraction.py`. Three behaviors:

1. **Symbol extraction** (`symbol` tag) — given source in
   language X, extraction produces chunks with correct
   `(kind, name, scope)`.
2. **Import extraction** (`import`/`multi_import` tags) — given
   source with imports, extraction produces import chunks with
   correct `ImportMeta` metadata.
3. **Mixed extraction** (`mixed` tag) — realistic modules produce
   all expected chunk kinds and method scoping.

Cross-cutting **properties** (determinism, line numbers, empty
source, error recovery) are tested via a parametrized fixture
over all languages with grammars.

Language-specific **edge cases** (java constructors, rust impl,
markdown heading exclusion, etc.) are plain test functions at the
bottom.
"""

from __future__ import annotations

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.index.models import ChunkKind, ImportMeta
from rbtr.index.treesitter import extract_symbols
from rbtr.languages import LanguageManager

from .conftest import extract_chunks

# ── Symbol extraction ────────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, expected",
    cases=".cases_extraction",
    has_tag="symbol",
)
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_chunks(lang, source)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


# ── Mixed extraction ─────────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, expected_kinds, expected_methods",
    cases=".cases_extraction",
    has_tag="mixed",
)
def test_extracts_all_expected_kinds(
    lang: str,
    source: str,
    expected_kinds: set[str],
    expected_methods: list[tuple[str, str]],
) -> None:
    """Realistic source produces all expected chunk kinds and method scoping."""
    chunks = extract_chunks(lang, source)
    kinds = {c.kind for c in chunks}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(c.name, c.scope) for c in chunks if c.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


# ── Import extraction ────────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, expected",
    cases=".cases_extraction",
    has_tag="import",
)
def test_extracts_import_metadata(lang: str, source: str, expected: dict) -> None:
    """First import chunk has the expected metadata."""
    chunks = extract_chunks(lang, source)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) >= 1, f"no import chunks extracted from {source!r}"
    assert imports[0].metadata == ImportMeta(**expected)


@parametrize_with_cases(
    "lang, source, count, metadata_list",
    cases=".cases_extraction",
    has_tag="multi_import",
)
def test_extracts_multi_import(
    lang: str, source: str, count: int, metadata_list: list[dict]
) -> None:
    """Multiple imports have correct count and per-import metadata."""
    chunks = extract_chunks(lang, source)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == count
    for imp, expected in zip(imports, metadata_list, strict=True):
        assert imp.metadata == ImportMeta(**expected)


# ── Cross-language properties ────────────────────────────────────────


@pytest.fixture(
    params=[
        pytest.param(("python", "def f():\n    pass\n"), id="python"),
        pytest.param(("javascript", "function f() {}\n"), id="javascript"),
        pytest.param(("typescript", "function f(): void {}\n"), id="typescript"),
        pytest.param(("go", "package main\nfunc f() {}\n"), id="go"),
        pytest.param(("rust", "fn f() {}\n"), id="rust"),
        pytest.param(("java", "class C { void f() {} }\n"), id="java"),
        pytest.param(("bash", "f() { :; }\n"), id="bash"),
        pytest.param(("c", "void f(void) { }\n"), id="c"),
        pytest.param(("cpp", "void f() { }\n"), id="cpp"),
        pytest.param(("ruby", "def f\n  1\nend\n"), id="ruby"),
        pytest.param(("sql", "CREATE TABLE t (id INT);\n"), id="sql"),
    ]
)
def lang_and_source(request: pytest.FixtureRequest) -> tuple[str, str]:
    """Language ID paired with a minimal source snippet."""
    return request.param


@pytest.fixture
def lang(lang_and_source: tuple[str, str]) -> str:
    return lang_and_source[0]


@pytest.fixture
def minimal_source(lang_and_source: tuple[str, str]) -> str:
    return lang_and_source[1]


def test_empty_source_returns_empty(lang: str) -> None:
    """Empty source produces no chunks, regardless of language."""
    assert extract_chunks(lang, "") == []


def test_deterministic_chunk_ids(lang: str, minimal_source: str) -> None:
    """Extracting the same source twice produces identical chunk IDs."""
    c1 = extract_chunks(lang, minimal_source)
    c2 = extract_chunks(lang, minimal_source)
    assert [c.id for c in c1] == [c.id for c in c2]


def test_blob_sha_propagated(lang: str, minimal_source: str) -> None:
    """All chunks carry the blob_sha passed to extract_symbols."""
    chunks = extract_chunks(lang, minimal_source)
    assert len(chunks) >= 1
    assert all(c.blob_sha == "sha1" for c in chunks)


def test_non_import_chunks_have_empty_metadata(lang: str, minimal_source: str) -> None:
    """Non-import chunks always have empty metadata."""
    chunks = extract_chunks(lang, minimal_source)
    for c in chunks:
        if c.kind != ChunkKind.IMPORT:
            assert c.metadata == ImportMeta(), (
                f"{c.kind} chunk {c.name!r} has metadata {c.metadata}"
            )


def test_line_numbers_are_positive(lang: str, minimal_source: str) -> None:
    """All chunks have positive, 1-indexed line numbers."""
    chunks = extract_chunks(lang, minimal_source)
    for c in chunks:
        assert c.line_start >= 1, f"{c.name} has line_start={c.line_start}"
        assert c.line_end >= c.line_start, f"{c.name} has line_end < line_start"


def test_content_is_nonempty(lang: str, minimal_source: str) -> None:
    """All chunks have non-empty content."""
    chunks = extract_chunks(lang, minimal_source)
    for c in chunks:
        assert c.content, f"{c.kind} chunk {c.name!r} has empty content"


def test_syntax_error_still_extracts_valid_parts(lang: str, minimal_source: str) -> None:
    """Tree-sitter error recovery: valid symbols extracted even with trailing garbage."""
    broken = minimal_source + "\n\x00\x00INVALID{{{[[\n"
    chunks = extract_chunks(lang, broken)
    assert len(chunks) >= 1


# ── Language-specific edge cases ─────────────────────────────────────


def test_java_constructor_not_captured() -> None:
    """Java constructors use constructor_declaration, not method_declaration."""
    src = """\
class Foo {
    Foo() {}
}
"""
    chunks = extract_chunks("java", src)
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD and c.name == "Foo"]
    assert methods == []


def test_rust_impl_captures_struct_and_impl() -> None:
    """Both struct and impl produce class chunks for the same type."""
    src = """\
struct Svc {}
impl Svc {
    fn new() -> Self { Svc {} }
}
"""
    chunks = extract_chunks("rust", src)
    svc_classes = [c for c in chunks if c.kind == ChunkKind.CLASS and c.name == "Svc"]
    assert len(svc_classes) == 2  # struct + impl


def test_sql_procedure_not_extracted() -> None:
    """CREATE PROCEDURE yields no chunk.

    The tree-sitter-sql 0.3.11 grammar has no `create_procedure`
    node, so the statement parses to an ERROR and the query
    matches nothing. If a future grammar adds the node this test
    flags that the plugin should map it.
    """
    src = """\
CREATE PROCEDURE refresh()
LANGUAGE SQL
AS $$ DELETE FROM cache; $$;
"""
    assert extract_chunks("sql", src) == []


def test_sql_schema_qualified_table_name() -> None:
    """A schema-qualified table is named by the table, not the schema."""
    src = """\
CREATE TABLE app.users (id INT);
"""
    chunks = extract_chunks("sql", src)
    classes = [c for c in chunks if c.kind == ChunkKind.CLASS]
    assert len(classes) == 1
    assert classes[0].name == "users"


def test_bash_source_and_dot_extracted_as_imports() -> None:
    """source/. commands are captured as imports."""
    src = """\
source ./env.sh
. /etc/profile
"""
    chunks = extract_chunks("bash", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 2
    modules = {c.metadata.module for c in imports}
    assert modules == {"./env.sh", "/etc/profile"}


def test_anonymous_chunk_when_name_capture_missing(
    language_manager: LanguageManager,
) -> None:
    """Chunks get name='<anonymous>' when the query omits the name capture."""
    grammar = language_manager.load_grammar("python")
    assert grammar is not None
    query_no_name = "(function_definition) @function\n"
    src = b"""\
def hello():
    pass
"""
    chunks = list(extract_symbols("test.py", "sha1", src, grammar, query_no_name))
    assert len(chunks) >= 1
    assert chunks[0].name == "<anonymous>"


def test_py_module_variable_content_is_whole_statement() -> None:
    """A module-level VARIABLE chunk spans the whole statement, named by LHS."""
    src = """\
MAX_SIZE = 100
"""
    chunks = extract_chunks("python", src)
    variables = [c for c in chunks if c.kind == ChunkKind.VARIABLE]
    assert len(variables) == 1
    assert variables[0].name == "MAX_SIZE"
    assert variables[0].content.strip() == "MAX_SIZE = 100"


def test_py_function_local_not_captured_as_variable() -> None:
    """Assignments inside a function stay part of the function chunk."""
    src = """\
def f():
    tmp = 1
    return tmp
"""
    chunks = extract_chunks("python", src)
    assert [c for c in chunks if c.kind == ChunkKind.VARIABLE] == []


def test_py_class_attribute_not_captured_as_variable() -> None:
    """Class-body attributes stay part of the class chunk, not VARIABLE chunks."""
    src = """\
class Config:
    DEFAULT = 30
"""
    chunks = extract_chunks("python", src)
    assert [c for c in chunks if c.kind == ChunkKind.VARIABLE] == []


def test_py_tuple_unpacking_captured_as_variables() -> None:
    """Flat tuple-unpacking binds each target as its own VARIABLE chunk.

    Both names come from one statement (tree-sitter fans the destructuring
    into a match per identifier), and each chunk spans the whole statement.
    """
    src = """\
a, b = compute()
"""
    chunks = extract_chunks("python", src)
    variables = [c for c in chunks if c.kind == ChunkKind.VARIABLE]
    assert {c.name for c in variables} == {"a", "b"}
    assert all(c.content.strip() == "a, b = compute()" for c in variables)


def test_unknown_capture_name_ignored(
    language_manager: LanguageManager,
) -> None:
    """Captures not in _CAPTURE_KIND are silently skipped."""
    grammar = language_manager.load_grammar("python")
    assert grammar is not None
    query_unknown = """\
(function_definition
  name: (identifier) @_fn_name) @function
(class_definition) @unknown_thing
"""
    src = b"""\
def f():
    pass

class C:
    pass
"""
    chunks = list(extract_symbols("test.py", "sha1", src, grammar, query_unknown))
    kinds = {c.kind for c in chunks}
    assert "function" in kinds
    assert "class" not in kinds


# ── Chunker-specific edge cases ──────────────────────────────────────


def test_md_subsection_excluded_from_parent_content() -> None:
    """Markdown parent chunk excludes child section content."""
    src = """\
# Parent

Parent body.

## Child

Child body.
"""
    chunks = extract_chunks("markdown", src)
    parent = next(c for c in chunks if c.name == "Parent")
    assert "Child body" not in parent.content
    assert "Parent body" in parent.content


def test_md_headingless_paragraphs() -> None:
    """Markdown without headings falls back to plaintext chunking."""
    src = """\
First paragraph.

Second paragraph.
"""
    chunks = extract_chunks("markdown", src)
    assert len(chunks) >= 1
    assert all(c.kind == ChunkKind.RAW_CHUNK for c in chunks)


def test_rst_hierarchy_from_adornment_order() -> None:
    """RST reconstructs hierarchy from adornment character order."""
    src = """\
Top
===

Mid
---

Deep
^^^

Content.
"""
    chunks = extract_chunks("rst", src)
    deep = next(c for c in chunks if c.name == "Deep")
    # Deep is the final section; scope shows parent chain.
    assert deep.scope == "Top::Mid"


def test_rst_overline_headings() -> None:
    """RST overline headings produce correct scope."""
    src = """\
=====
Title
=====

Intro.

----------
Subsection
----------

Body.
"""
    chunks = extract_chunks("rst", src)
    sub = next(c for c in chunks if c.name == "Subsection")
    assert sub.scope == "Title"


def test_toml_dotted_key_scope() -> None:
    """TOML dotted keys split into name + scope."""
    src = """\
[tool.ruff]
line-length = 99
"""
    chunks = extract_chunks("toml", src)
    assert chunks[0].name == "ruff"
    assert chunks[0].scope == "tool"


def test_html_no_body_fallback() -> None:
    """HTML without body falls back to single chunk."""
    chunks = extract_chunks("html", "<div>hello</div>")
    assert len(chunks) == 1


def test_yaml_no_mapping_fallback() -> None:
    """YAML without mapping falls back to single chunk."""
    chunks = extract_chunks("yaml", "- item1\n- item2\n")
    assert len(chunks) == 1


def test_html_script_src_produces_import() -> None:
    """HTML <script src> produces an import chunk."""
    src = """\
<html>
<head><script src="app.js"></script></head>
<body><p>hello</p></body>
</html>
"""
    chunks = extract_chunks("html", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "app.js"
    assert imports[0].metadata.language_hint == "javascript"


def test_html_link_href_produces_import() -> None:
    """HTML <link href> produces an import chunk."""
    src = """\
<html>
<head><link rel="stylesheet" href="styles.css"></head>
<body><p>hello</p></body>
</html>
"""
    chunks = extract_chunks("html", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "styles.css"
    assert imports[0].metadata.language_hint == "css"


def test_css_import_produces_import_chunk() -> None:
    """CSS @import url(...) produces an import chunk."""
    src = """\
@import url("reset.css");
body { color: #333; }
"""
    chunks = extract_chunks("css", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "reset.css"


# ── Markdown link extraction ──────────────────────────────────────────


def test_md_local_link_produces_import() -> None:
    """Markdown [text](local.md) produces an IMPORT chunk."""
    src = """\
# Guide

See [other doc](other.md) for details.
"""
    chunks = extract_chunks("markdown", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "other.md"


def test_md_link_with_fragment_sets_names() -> None:
    """Markdown [text](path.md#section) sets module and names."""
    src = """\
# Guide

See [API section](api.md#my-function) for the API.
"""
    chunks = extract_chunks("markdown", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "api.md"
    assert imports[0].metadata.names == "my-function"


def test_md_relative_link_produces_import() -> None:
    """Markdown [text](../src/foo.py) with relative path produces import."""
    src = """\
# Docs

See [source](../src/foo.py) for details.
"""
    chunks = extract_chunks("markdown", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "../src/foo.py"


def test_md_external_link_skipped() -> None:
    """Markdown links to external URLs produce no import chunk."""
    src = """\
# Guide

See [example](https://example.com) and [mail](mailto:a@b.com).
"""
    chunks = extract_chunks("markdown", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []


def test_md_fragment_only_link_skipped() -> None:
    """Markdown #-only links (same-file anchors) produce no import."""
    src = """\
# Guide

See [below](#details) for more.
"""
    chunks = extract_chunks("markdown", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []


def test_md_multiple_links_in_one_section() -> None:
    """Multiple links in a single section produce multiple imports."""
    src = """\
# Guide

See [a](one.md) and [b](two.py) here.
"""
    chunks = extract_chunks("markdown", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    modules = {c.metadata.module for c in imports}
    assert modules == {"one.md", "two.py"}


def test_md_bare_mention_no_import() -> None:
    """Prose mentioning a symbol name without a link produces no import."""
    src = """\
# Guide

Call do_stuff to process the data.
"""
    chunks = extract_chunks("markdown", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []


# ── RST reference extraction ───────────────────────────────────────────


def test_rst_func_role_produces_import() -> None:
    """RST :func:`name` produces IMPORT with names field."""
    src = """\
Title
=====

See :func:`do_stuff` for details.
"""
    chunks = extract_chunks("rst", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.names == "do_stuff"
    assert imports[0].metadata.module == ""


def test_rst_class_role_produces_import() -> None:
    """RST :class:`Name` produces IMPORT with names field."""
    src = """\
Title
=====

See :class:`User` for the model.
"""
    chunks = extract_chunks("rst", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.names == "User"


def test_rst_meth_role_produces_import() -> None:
    """RST :meth:`Class.method` produces IMPORT with names field."""
    src = """\
Title
=====

See :meth:`User.save` for persistence.
"""
    chunks = extract_chunks("rst", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.names == "User.save"


def test_rst_func_tilde_strips_prefix() -> None:
    """RST :func:`~module.name` strips ~ and uses last component."""
    src = """\
Title
=====

See :func:`~mypackage.utils.do_stuff` for details.
"""
    chunks = extract_chunks("rst", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.names == "do_stuff"


def test_rst_doc_role_produces_import() -> None:
    """RST :doc:`path` produces IMPORT with module field."""
    src = """\
Title
=====

See :doc:`api/module` for the API.
"""
    chunks = extract_chunks("rst", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "api/module"


def test_rst_mod_role_produces_import() -> None:
    """RST :mod:`name` produces IMPORT with module field."""
    src = """\
Title
=====

See :mod:`mypackage.utils` for helpers.
"""
    chunks = extract_chunks("rst", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "mypackage.utils"


def test_rst_toctree_produces_imports() -> None:
    """RST toctree directive produces one IMPORT per entry."""
    src = """\
Title
=====

.. toctree::
   :maxdepth: 2

   intro
   api/index
"""
    chunks = extract_chunks("rst", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    modules = {c.metadata.module for c in imports}
    assert modules == {"intro", "api/index"}


def test_rst_reference_local_produces_import() -> None:
    """RST `text <url>`_ with local path produces import."""
    src = """\
Title
=====

See `the guide <other.rst>`_ for details.
"""
    chunks = extract_chunks("rst", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "other.rst"


def test_rst_reference_external_skipped() -> None:
    """RST `text <url>`_ with external URL produces no import."""
    src = """\
Title
=====

See `example <https://example.com>`_ for details.
"""
    chunks = extract_chunks("rst", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []


def test_rst_plain_prose_no_import() -> None:
    """RST prose mentioning a symbol without a role produces no import."""
    src = """\
Title
=====

Call do_stuff to process the data.
"""
    chunks = extract_chunks("rst", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []
