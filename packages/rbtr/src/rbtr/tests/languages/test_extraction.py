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

The empty-source **property** is checked here for every registered
language; the remaining cross-language invariants (determinism, line
numbers, non-empty content, metadata, error recovery) run over the
committed samples in `test_samples.py`.

Language-specific **edge cases** (java constructors, rust impl,
markdown heading exclusion, etc.) are plain test functions at the
bottom.
"""

from __future__ import annotations

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.index.models import ChunkKind, ImportMeta
from rbtr.index.treesitter import extract_symbols
from rbtr.languages import LanguageManager, get_manager

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


@pytest.mark.parametrize("lang", sorted(get_manager().all_language_ids()))
def test_empty_source_yields_host_presence(lang: str) -> None:
    """Empty source yields one content-less host-presence chunk, for every
    registered language.

    Records the file's host language so the blob-dedup gate skips an empty
    file on later builds instead of re-parsing it every time. The remaining
    cross-language invariants (determinism, line numbers, content, metadata,
    error recovery) run over the committed samples in `test_samples.py`.
    """
    chunks = extract_chunks(lang, "")
    assert len(chunks) == 1
    assert chunks[0].content == ""
    assert chunks[0].language == lang


# ── Language-specific edge cases ─────────────────────────────────────


def test_sql_pragma_not_extracted() -> None:
    """A DuckDB PRAGMA yields no definition chunk.

    The grammar has no PRAGMA statement node, so it parses to a
    top-level ERROR with no enclosing `statement` to capture. This
    is a known limitation guard; it flags the day the grammar gains
    PRAGMA support. Only the content-less host-presence chunk (for
    blob dedup) remains.
    """
    src = "PRAGMA create_fts_index('chunks', 'id', 'body');\n"
    chunks = extract_chunks("sql", src)
    assert [c for c in chunks if c.content] == []


def test_sql_multi_statement_one_chunk_each() -> None:
    """Each top-level statement in a file becomes its own chunk."""
    src = """\
CREATE TABLE a (id INT);
SELECT * FROM a;
DROP TABLE a;
"""
    chunks = extract_chunks("sql", src)
    assert [(c.kind, c.name) for c in chunks] == [
        (ChunkKind.CLASS, "a"),
        (ChunkKind.FUNCTION, "a"),
        (ChunkKind.FUNCTION, "a"),
    ]


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


def test_scope_extractor_owns_scope_address(
    language_manager: LanguageManager,
) -> None:
    """A `scope_extractor` overrides the default scope with its own segments."""
    grammar = language_manager.load_grammar("python")
    assert grammar is not None
    query = "(function_definition name: (identifier) @_fn_name) @function\n"
    src = b"def hello():\n    pass\n"
    chunks = list(
        extract_symbols(
            "test.py",
            "sha1",
            src,
            grammar,
            query,
            scope_extractor=lambda _cap, _node, _caps: ["a", "b"],
        )
    )
    assert len(chunks) == 1
    assert chunks[0].scope == "a::b"


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


def test_md_chunker_target_extracted() -> None:
    """A chunker-based target (yaml) inside a Markdown fence extracts.

    Query targets (python) already worked; this proves the delegate runs a
    chunker plugin over the block too, at absolute line numbers.
    """
    src = """\
# Doc

```yaml
service: greeter
```
"""
    chunks = extract_chunks("markdown", src)
    yaml_sections = [c for c in chunks if c.language == "yaml"]
    assert [c.name for c in yaml_sections] == ["service"]
    assert yaml_sections[0].line_start == 4


def test_md_unknown_fence_left_unparsed() -> None:
    """A fence naming a language rbtr has no grammar for is not delegated."""
    src = """\
# Doc

```nonexistent
some content here
```
"""
    chunks = extract_chunks("markdown", src)
    assert {c.language for c in chunks} == {"markdown"}


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
    """A TOML dotted table splits into last-segment name + preceding scope."""
    src = """\
[tool.ruff.lint]
select = ["E"]
"""
    chunks = extract_chunks("toml", src)
    assert chunks[0].name == "lint"
    assert chunks[0].scope == "tool::ruff"


def test_html_non_semantic_only_emits_presence() -> None:
    """HTML with no head/body/semantic element yields only a presence chunk.

    A non-semantic wrapper (`<div>`) is not elevated, so no searchable
    doc_section is produced; the engine appends one content-less host chunk.
    """
    chunks = extract_chunks("html", "<div>hello</div>")
    assert [c.kind for c in chunks if c.content] == []
    assert [c.language for c in chunks] == ["html"]


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


def test_html_self_closing_link_produces_import() -> None:
    """An XHTML-style self-closing `<link ... />` produces an import."""
    src = """\
<html>
<head><link rel="stylesheet" href="styles.css" /></head>
<body><p>hello</p></body>
</html>
"""
    chunks = extract_chunks("html", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 1
    assert imports[0].metadata.module == "styles.css"


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
