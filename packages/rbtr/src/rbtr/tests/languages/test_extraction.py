"""Engine-level extraction tests.

Per-language extraction lives in each `rbtr-lang-*` package. These tests pin
the language-agnostic engine mechanics: the name/scope resolver fallbacks and
capture-kind filtering in `extract_symbols` (exercised with synthetic
registrations over the Python grammar), comment grouping/folding, and the
content-less host-presence chunk emitted for every registered language.
"""

from __future__ import annotations

import polars as pl
import pytest
from tree_sitter import Parser

from rbtr.config import config
from rbtr.domain.models import ChunkKind, Fragment, Import
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_batch, extract_file, extract_symbols, file_rows
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration, QueryExtraction
from rbtr.languages.treesitter import runs


@pytest.mark.parametrize("lang", sorted(get_manager().all_language_ids()))
def test_empty_source_yields_host_presence(lang: str) -> None:
    """Empty source yields one content-less host-presence chunk, for every
    registered language.

    Records the file's host language so the blob-dedup gate skips an empty
    file on later builds instead of re-parsing it every time.
    """
    chunks = extract_file(FileEntry("input", "sha1", b""), lang)
    assert len(chunks) == 1
    assert chunks[0].content == ""
    assert chunks[0].language == lang


def test_chunk_is_unnamed_when_name_capture_missing() -> None:
    """Chunks get an empty name when the query omits the name capture."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query_no_name = "(function_definition) @function\n"
    src = b"""\
def hello():
    pass
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query_no_name))
    chunks = list(extract_symbols(reg, "test.py", "sha1", src, grammar))
    assert len(chunks) >= 1
    assert chunks[0].name == ""


def test_scope_extractor_owns_scope_address() -> None:
    """A `scope_extractor` overrides the default scope with its own segments."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = "(function_definition name: (identifier) @_fn_name) @function\n"
    src = b"def hello():\n    pass\n"
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    reg.scope_extractor(lambda _resolver, _cap, _node, _caps: ["a", "b"])
    chunks = list(extract_symbols(reg, "test.py", "sha1", src, grammar))
    assert len(chunks) == 1
    assert chunks[0].scope == "a::b"


def test_an_enclosing_scope_is_named_by_its_own_capture() -> None:
    """A scope node with no `name` field takes the name its capture gave it.

    A Markdown `section` has no `name` field; its heading names it.
    """
    grammar = get_manager().grammar("markdown")
    assert grammar is not None
    query = "(section (atx_heading (inline) @_section_name)) @doc_section\n"
    reg = LanguageRegistration(
        id="faketest",
        extraction=QueryExtraction(query=query, scope_types=frozenset({"section"})),
    )
    src = b"# Title\n\n## Install\n\n### Pip\n\npip.\n\n## Use\n\nCall it.\n"
    chunks = list(extract_symbols(reg, "doc.md", "sha1", src, grammar))
    assert [(c.role.name, c.scope) for c in chunks if c.role.kind == ChunkKind.DOC_SECTION] == [
        ("Title", ""),
        ("Install", "Title"),
        ("Pip", "Title::Install"),
        ("Use", "Title"),
    ]


def test_a_captured_reference_names_its_fragment() -> None:
    """`path#fragment` names `fragment` in `path`; a fragment alone names it in the file itself."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = """\
(call
  function: (identifier) @_f
  arguments: (argument_list (string (string_content) @_import_module))
  (#eq? @_f "load")) @import
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    src = b'load("page.html#intro")\nload("#usage")\nload("plain.html")\n'
    chunks = list(extract_symbols(reg, "docs/guide.html", "sha1", src, grammar))
    assert [
        (c.role.meta.module, c.role.meta.names) for c in chunks if isinstance(c.role, Import)
    ] == [("page.html", "intro"), ("guide.html", "usage"), ("plain.html", "")]


def test_a_section_ends_where_its_first_subsection_begins() -> None:
    """A captured section's chunk holds its own text; each subsection is a chunk of its own."""
    grammar = get_manager().grammar("markdown")
    assert grammar is not None
    query = "(section (atx_heading (inline) @_section_name)) @doc_section\n"
    reg = LanguageRegistration(
        id="faketest",
        extraction=QueryExtraction(query=query, scope_types=frozenset({"section"})),
    )
    src = b"# Title\n\nIntro.\n\n## Install\n\nRun it.\n"
    chunks = list(extract_symbols(reg, "doc.md", "sha1", src, grammar))
    assert [(c.role.name, c.content, c.span.start, c.span.end) for c in chunks] == [
        ("Title", "# Title\n\nIntro.", 1, 3),
        ("Install", "## Install\n\nRun it.", 5, 7),
    ]


def test_a_long_left_over_node_is_split_into_its_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node no capture covers, longer than `chunk_lines`, becomes fragments of its children."""
    monkeypatch.setattr(config, "chunk_lines", 4)
    grammar = get_manager().grammar("markdown")
    assert grammar is not None
    query = "(section (atx_heading (inline) @_section_name)) @doc_section\n"
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    src = b"One.\n\nTwo.\n\nThree.\n\nFour.\n"
    chunks = list(extract_symbols(reg, "doc.md", "sha1", src, grammar))
    assert [(c.content, c.span.start) for c in chunks if isinstance(c.role, Fragment)] == [
        ("One.", 1),
        ("Two.", 3),
        ("Three.", 5),
        ("Four.", 7),
    ]


def test_strip_directive_strips_a_captured_name() -> None:
    """`#strip!` removes what its regex matches from the name a capture gives."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = """\
((function_definition name: (identifier) @_fn_name) @function
  (#strip! @_fn_name "^_+"))
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    chunks = list(extract_symbols(reg, "a.py", "sha1", b"def __hidden():\n    pass\n", grammar))
    assert [c.role.name for c in chunks if c.role.kind == ChunkKind.FUNCTION] == ["hidden"]


def test_strip_directive_shapes_an_import() -> None:
    """`#strip!` on `@_import_module` and `@_import_names` shapes the import's module and names."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = """\
((call
  function: (identifier) @_f
  arguments: (argument_list
    (string (string_content) @_import_module)
    (string (string_content) @_import_names))) @import
  (#eq? @_f "ref")
  (#strip! @_import_module "^pkg:")
  (#strip! @_import_names "^~"))
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    src = b'ref("pkg:guide", "~intro")\n'
    [meta] = [
        c.role.meta
        for c in extract_symbols(reg, "a.py", "sha1", src, grammar)
        if isinstance(c.role, Import)
    ]
    assert (meta.module, meta.names) == ("guide", "intro")


def test_a_quantified_import_capture_names_one_module_per_line() -> None:
    """Nodes of a quantified `@_import_module` make one import per line they sit on.

    For a grammar that gives a list's entries no node of their own; each
    line's import is named and spanned by that line, its `#strip!` applied.
    """
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = """\
((list ((string (string_content) @_import_module) ","?)+) @import
  (#strip! @_import_module "^x:"))
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    src = b'LINKS = [\n    "x:topics/intro",\n    "x:api/index",\n]\n'
    imports = [
        (c.role, c.span)
        for c in extract_symbols(reg, "a.py", "sha1", src, grammar)
        if isinstance(c.role, Import)
    ]
    assert [(role.name, role.meta.module, span.start, span.end) for role, span in imports] == [
        ("x:topics/intro", "topics/intro", 2, 2),
        ("x:api/index", "api/index", 3, 3),
    ]


def test_flat_sections_are_levelled_by_their_level_captures() -> None:
    """Sections a grammar leaves flat nest by `@_level`, ranked by first appearance.

    Each holds its text up to the next such section, under the nearest
    earlier one of a higher level. An overlined `=` is a style of its own.
    """
    grammar = get_manager().grammar("rst")
    assert grammar is not None
    query = """\
((section "adornment"? @_level (title) @_section_name "adornment" @_level
  (#strip! @_level "(?<=^.).*")) @doc_section)
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    src = b"""\
=====
Guide
=====

Intro text.

Install
=======

Run it.

Pip
---

pip.

Use
===

Call it.
"""
    sections = [
        (c.role.name, c.scope, c.span.start, c.span.end)
        for c in extract_symbols(reg, "doc.rst", "sha1", src, grammar)
        if c.role.kind == ChunkKind.DOC_SECTION
    ]
    assert sections == [
        ("Guide", "", 1, 5),
        ("Install", "Guide", 7, 10),
        ("Pip", "Guide::Install", 12, 15),
        ("Use", "Guide", 17, 20),
    ]


def test_set_directive_names_where_an_import_finds_its_name() -> None:
    """`#set! import.language_hint` says which language's chunks hold the name, as a module would."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = """\
((call function: (identifier) @_f arguments: (argument_list (string (string_content) @_import_names))) @import
  (#set! import.language_hint "rst")
  (#eq? @_f "ref"))
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    [meta] = [
        c.role.meta
        for c in extract_symbols(reg, "a.py", "sha1", b'ref("intro")\n', grammar)
        if isinstance(c.role, Import)
    ]
    assert (meta.names, meta.language_hint) == ("intro", "rst")


def test_unknown_capture_name_ignored() -> None:
    """Captures not in CAPTURE_KINDS are silently skipped."""
    grammar = get_manager().grammar("python")
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
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query_unknown))
    chunks = list(extract_symbols(reg, "test.py", "sha1", src, grammar))
    kinds = {c.kind for c in chunks}
    assert "function" in kinds
    assert "class" not in kinds


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        pytest.param(
            b"# one\n# two\n# three\n",
            [(1, 3, "# one\n# two\n# three")],
            id="contiguous_run_is_one_ordered_block",
        ),
        pytest.param(
            b"# a1\n# a2\n\n# b1\n# b2\n",
            [(1, 2, "# a1\n# a2"), (4, 5, "# b1\n# b2")],
            id="blank_line_splits_into_two_source_ordered_blocks",
        ),
        pytest.param(
            b"# doc\ndef f():\n    pass\n\n\n# tail\n",
            [(6, 6, "# tail")],
            id="code_splits_run_and_leading_block_folds_into_symbol",
        ),
    ],
)
def test_comment_blocks_group_in_source_order(
    src: bytes, expected: list[tuple[int, int, str]]
) -> None:
    """Comment runs group into blocks whose lines stay in source order.

    A blank-free run is one block with its lines contiguous and ordered; a
    blank line splits a run into separate blocks emitted in source order; code
    between comments splits them (and a block flush before a symbol folds into
    it rather than standing alone).
    """
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = "(comment) @comment\n(function_definition name: (identifier) @_fn_name) @function\n"
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    chunks = list(extract_symbols(reg, "t.py", "sha1", src, grammar))
    comment_chunks = [c for c in chunks if c.kind == ChunkKind.COMMENT]
    comments = [(c.line_start, c.line_end, c.content) for c in comment_chunks]
    assert comments == expected
    assert [c.name for c in comment_chunks] == [""] * len(comment_chunks), (
        "a comment has no identifier to name it"
    )


def test_trailing_comment_does_not_fold_into_next_symbol() -> None:
    """A comment trailing code on its line is not the next symbol's doc."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = "(comment) @comment\n(module (expression_statement (assignment left: (identifier) @_var_name) @variable))\n"
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    src = b"x = 1  # trailing about x\ny = 2\n"
    chunks = list(extract_symbols(reg, "t.py", "sha1", src, grammar))
    y = next(c for c in chunks if c.name == "y")
    assert "trailing about x" not in y.content
    assert any(c.kind == ChunkKind.COMMENT and "trailing about x" in c.content for c in chunks)


def test_trailing_comment_not_grouped_with_following_leading_doc() -> None:
    """A trailing comment is separate from a real own-line doc for the next symbol."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = "(comment) @comment\n(function_definition name: (identifier) @_fn_name) @function\n"
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    src = b"x = 1  # trailing\n# real doc\ndef f():\n    pass\n"
    f = next(c for c in extract_symbols(reg, "t.py", "sha1", src, grammar) if c.name == "f")
    assert "# real doc" in f.content
    assert "# trailing" not in f.content


@pytest.mark.parametrize(
    ("query", "src", "expected"),
    [
        pytest.param(
            "(import_statement) @import\n(function_definition name: (identifier) @_fn_name) @function\n",
            b"""\
import os

setup()
configure()

def f():
    pass

if __name__ == "__main__":
    f()
""",
            [
                (3, 4, "setup()\nconfigure()", ("expression_statement",)),
                (9, 10, 'if __name__ == "__main__":\n    f()', ("if_statement",)),
            ],
            id="statements_join_until_a_blank_line_or_a_definition",
        ),
        pytest.param(
            "(import_statement) @import\n",
            b"""\
if TYPE_CHECKING:
    import os
""",
            [(1, 1, "if TYPE_CHECKING:", ("identifier",))],
            id="a_partly_covered_statement_leaves_only_its_own_lines",
        ),
        pytest.param(
            "(function_definition name: (identifier) @_fn_name) @function\n",
            b"x = (\n",
            [(1, 1, "x = (", ("ERROR",))],
            id="a_parse_error_is_a_fragment",
        ),
    ],
)
def test_code_no_pattern_names_becomes_fragments(
    query: str, src: bytes, expected: list[tuple[int, int, str, tuple[str, ...]]]
) -> None:
    """Every line a query leaves reaches a fragment, recorded with the syntax it holds."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    chunks = list(extract_symbols(reg, "t.py", "sha1", src, grammar))
    fragments = [
        (c.line_start, c.line_end, c.content, c.role.meta.node_types)
        for c in chunks
        if isinstance(c.role, Fragment)
    ]
    assert fragments == expected
    assert {c.language for c in chunks if c.kind == ChunkKind.FRAGMENT} <= {"faketest"}


def test_a_fragment_is_scoped_to_the_scopes_around_it() -> None:
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = "(function_definition name: (identifier) @_fn_name) @function\n"
    src = b"""\
class Config:
    retries = 3

    def load(self):
        pass
"""
    reg = LanguageRegistration(
        id="faketest",
        extraction=QueryExtraction(query=query, scope_types=frozenset({"class_definition"})),
    )
    chunks = list(extract_symbols(reg, "t.py", "sha1", src, grammar))
    fragments = [(c.content, c.scope) for c in chunks if c.kind == ChunkKind.FRAGMENT]
    assert fragments == [("class Config:", "Config"), ("retries = 3", "Config")]


def test_a_run_of_code_splits_at_the_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "chunk_lines", 2)
    grammar = get_manager().grammar("python")
    assert grammar is not None
    reg = LanguageRegistration(
        id="faketest", extraction=QueryExtraction(query="(comment) @comment\n")
    )
    src = b"""\
a()
b()
c()
"""
    chunks = list(extract_symbols(reg, "t.py", "sha1", src, grammar))
    assert [(c.line_start, c.line_end) for c in chunks] == [(1, 2), (3, 3)]


def test_a_block_left_to_its_injection_is_not_a_fragment() -> None:
    """An HTML `<script>`'s content belongs to the javascript its injection extracts; a tag on a line of its own is the page's."""
    manager = get_manager()
    html = manager.get_registration("html")
    grammar = manager.grammar("html")
    assert html is not None
    assert grammar is not None
    src = b"""\
<p>intro</p>
<script>
run();
</script>
"""
    chunks = list(extract_symbols(html, "page.html", "sha1", src, grammar))
    fragments = [
        (c.line_start, c.line_end, c.content) for c in chunks if c.kind == ChunkKind.FRAGMENT
    ]
    assert fragments == [(1, 1, "<p>intro</p>"), (4, 4, "</script>")]


def test_a_callee_becomes_the_path_it_names() -> None:
    """Names and attribute paths are paths; one starting at a `self` or `super` marker reads from it.

    A callee through a value (`x[0].m()`, `f().g()`) is no path, and a
    marker alone (`super()`) names nothing.
    """
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = """\
(function_definition name: (identifier) @_fn_name) @function
(call function: (_) @_call)
((identifier) @_self (#eq? @_self "self"))
((identifier) @_super (#eq? @_super "super"))
"""
    src = b"""\
def run(self):
    foo()
    a.b.c()
    self.save()
    super().__init__()
    x[0].m()
    f().g()
    foo()
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    rows = extract_batch(
        file_rows([(FileEntry("t.py", "sha1", src), reg.id)]), {reg.id: (reg, grammar)}.get
    )
    assert rows.filter(pl.col("name") == "run")["metadata"].struct.field("refs").to_list() == [
        [
            {"kind": "calls", "lookup": "unqualified", "path": ["foo"], "type": None},
            {"kind": "calls", "lookup": "unqualified", "path": ["a", "b", "c"], "type": None},
            {"kind": "calls", "lookup": "member", "path": ["save"], "type": None},
            {"kind": "calls", "lookup": "base", "path": ["__init__"], "type": None},
            {"kind": "calls", "lookup": "unqualified", "path": ["f"], "type": None},
        ]
    ]


def test_a_reference_belongs_to_the_innermost_chunk_around_it() -> None:
    """A method's call is the method's, a class body's the class's, a module-level call a fragment's.

    A base is the class's own; `metaclass=Meta` is no base the query captures.
    """
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = """\
(function_definition name: (identifier) @_fn_name) @function
(class_definition name: (identifier) @_cls_name) @class
(call function: (_) @_call)
(class_definition superclasses: (argument_list [(identifier) (attribute)] @_base))
"""
    src = b"""\
class Box(Base, mod.Mixin, metaclass=Meta):
    size = compute()

    def open(self):
        helper()


helper()
"""
    reg = LanguageRegistration(
        id="faketest",
        extraction=QueryExtraction(query=query, scope_types=frozenset({"class_definition"})),
    )
    rows = extract_batch(
        file_rows([(FileEntry("t.py", "sha1", src), reg.id)]), {reg.id: (reg, grammar)}.get
    )
    assert rows.select(
        pl.col("kind").cast(pl.String),
        "name",
        refs=pl.col("metadata").struct.field("refs"),
    ).rows() == [
        (
            "class",
            "Box",
            [
                {"kind": "inherits", "lookup": "unqualified", "path": ["Base"], "type": None},
                {
                    "kind": "inherits",
                    "lookup": "unqualified",
                    "path": ["mod", "Mixin"],
                    "type": None,
                },
                {"kind": "calls", "lookup": "unqualified", "path": ["compute"], "type": None},
            ],
        ),
        (
            "method",
            "open",
            [{"kind": "calls", "lookup": "unqualified", "path": ["helper"], "type": None}],
        ),
        (
            "fragment",
            "",
            [{"kind": "calls", "lookup": "unqualified", "path": ["helper"], "type": None}],
        ),
    ]


def test_a_call_through_a_functions_local_is_on_a_value() -> None:
    """A call whose head is a local of a function around it is on a value, with the local's written type.

    A nested function sees its enclosing function's locals and a sibling
    does not; a definition of the name inside the function wins over the
    local; a class body holds no locals.
    """
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = """\
(function_definition name: (identifier) @_fn_name) @function
(class_definition name: (identifier) @_cls_name) @class
(call function: (_) @_call)
(parameters (identifier) @_local)
(parameters (typed_parameter (identifier) @_local type: (type) @_local_type))
(assignment left: (identifier) @_local)
"""
    src = b"""\
def run(order: Order, f, g):
    f()
    order.save()
    g()

    def g():
        pass

    def inner():
        f()


def other():
    f()


class Box:
    size = 1

    def open(self):
        size()
"""
    reg = LanguageRegistration(
        id="faketest",
        extraction=QueryExtraction(query=query, scope_types=frozenset({"class_definition"})),
    )
    rows = extract_batch(
        file_rows([(FileEntry("t.py", "sha1", src), reg.id)]), {reg.id: (reg, grammar)}.get
    )
    assert rows.filter(pl.col("kind") != ChunkKind.FRAGMENT).select(
        pl.col("kind").cast(pl.String),
        "name",
        refs=pl.col("metadata").struct.field("refs"),
    ).rows() == [
        (
            "function",
            "run",
            [
                {"kind": "calls", "lookup": "type", "path": ["f"], "type": None},
                {
                    "kind": "calls",
                    "lookup": "type",
                    "path": ["order", "save"],
                    "type": ["Order"],
                },
                {"kind": "calls", "lookup": "unqualified", "path": ["g"], "type": None},
            ],
        ),
        ("function", "g", []),
        (
            "function",
            "inner",
            [{"kind": "calls", "lookup": "type", "path": ["f"], "type": None}],
        ),
        (
            "function",
            "other",
            [{"kind": "calls", "lookup": "unqualified", "path": ["f"], "type": None}],
        ),
        ("class", "Box", []),
        (
            "method",
            "open",
            [{"kind": "calls", "lookup": "unqualified", "path": ["size"], "type": None}],
        ),
    ]


def test_a_super_call_calls_the_bases_method_of_its_functions_name() -> None:
    """A `@_super_call` (Ruby's bare `super`) calls the bases' method of its function's own name.

    In a class body it names nothing, so it is no reference.
    """
    grammar = get_manager().grammar("ruby")
    assert grammar is not None
    query = """\
(method name: (identifier) @_fn_name) @function
(class name: (constant) @_cls_name) @class
(super) @_super_call
"""
    src = b"""\
class Order < Model
  super
  def save
    super
  end
end
"""
    reg = LanguageRegistration(
        id="faketest", extraction=QueryExtraction(query=query, scope_types=frozenset({"class"}))
    )
    rows = extract_batch(
        file_rows([(FileEntry("t.rb", "sha1", src), reg.id)]), {reg.id: (reg, grammar)}.get
    )
    assert rows.filter(pl.col("kind") != ChunkKind.FRAGMENT).select(
        pl.col("kind").cast(pl.String), "name", refs=pl.col("metadata").struct.field("refs")
    ).rows() == [
        ("class", "Order", []),
        (
            "method",
            "save",
            [{"kind": "calls", "lookup": "base", "path": ["save"], "type": None}],
        ),
    ]


def test_a_callee_captured_in_parts_is_one_path() -> None:
    """Nodes one match captures as `@_call` are one span: a grammar with no single callee node captures each part."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = """\
(function_definition name: (identifier) @_fn_name) @function
(call function: (attribute object: (identifier) @_call attribute: (identifier) @_call))
"""
    src = b"""\
def run():
    models . Field()
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    rows = extract_batch(
        file_rows([(FileEntry("t.py", "sha1", src), reg.id)]), {reg.id: (reg, grammar)}.get
    )
    assert rows.filter(pl.col("name") == "run")["metadata"].struct.field("refs").to_list() == [
        [{"kind": "calls", "lookup": "unqualified", "path": ["models", "Field"], "type": None}]
    ]


@pytest.mark.parametrize(
    ("query", "src", "expected"),
    [
        pytest.param(
            """\
(import_from_statement
  module_name: (dotted_name) @_import_module
  name: (dotted_name) @_import_names) @import
""",
            b"from m import a, b\n",
            [{"module": "m", "names": "a,b", "bindings": []}],
            id="a-list-of-names",
        ),
        pytest.param(
            """\
(import_from_statement
  module_name: (dotted_name) @_import_module
  name: (dotted_name) @_import_names @_bind_name) @import

(import_from_statement
  module_name: (dotted_name) @_import_module
  name: (aliased_import
    name: (dotted_name) @_import_names @_bind_name
    alias: (identifier) @_bind_local)) @import
""",
            b"from m import a, b as c\n",
            [
                {
                    "module": "m",
                    "names": "a,b",
                    "bindings": [{"name": "a", "local": "a"}, {"name": "b", "local": "c"}],
                }
            ],
            id="matches-on-one-import-merge",
        ),
        pytest.param(
            """\
(import_statement
  name: (aliased_import
    name: (dotted_name) @_import_module
    alias: (identifier) @_bind_local)) @import
""",
            b"import numpy as np\n",
            [{"module": "numpy", "names": "", "bindings": [{"name": "numpy", "local": "np"}]}],
            id="a-local-alone-binds-the-module",
        ),
        pytest.param(
            """\
(import_statement name: (dotted_name) @_import_module) @import
""",
            b"import os, sys\n",
            [
                {"module": "os", "names": "", "bindings": []},
                {"module": "sys", "names": "", "bindings": []},
            ],
            id="two-modules-are-two-imports",
        ),
        pytest.param(
            """\
((import_from_statement
  module_name: (dotted_name) @_import_module
  (wildcard_import))
  (#set! import.names "*")) @import
""",
            b"from m import *\n",
            [{"module": "m", "names": "*", "bindings": []}],
            id="a-pattern-sets-names",
        ),
        pytest.param(
            """\
((import_statement name: (dotted_name) @_import_module)
  (#set! import.prefix "./")) @import
""",
            b"import helpers\n",
            [{"module": "./helpers", "names": "", "bindings": []}],
            id="a-pattern-writes-a-prefix-before-its-module",
        ),
    ],
)
def test_a_query_states_an_imports_names_and_bindings(
    query: str, src: bytes, expected: list[dict]
) -> None:
    """Names, bindings and directives come from captures alone, with no import extractor."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    imports = [
        c.role.meta.model_dump(include={"module", "names", "bindings"})
        for c in extract_symbols(reg, "t.py", "sha1", src, grammar)
        if isinstance(c.role, Import)
    ]
    assert imports == expected


def test_a_partly_covered_leaf_is_a_run_of_its_own() -> None:
    """A leaf spanning lines, one of them covered, has no children to look into: it is its own run."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    src = b'x = """a\nb\nc"""\n'
    tree = Parser(grammar).parse(src)
    covered = bytearray(src.count(b"\n") + 2)
    covered[1] = 1
    assert [[node.type for node in run] for run in runs(tree.root_node, covered, src)] == [
        ["string_content"]
    ]
