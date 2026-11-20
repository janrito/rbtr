"""Engine-level extraction tests, through the engine.

Per-language extraction lives in each `rbtr-lang-*` package. These tests pin
the language-agnostic engine mechanics: the name and scope fallbacks,
capture-kind filtering and query directives (with synthetic registrations
over installed grammars), comment grouping and folding, fragments, and the
content-less host-presence chunk of every language a file can be. Each
builds one file with `indexed`, only its registration loaded
(`limited_to`), and reads its chunks back with `Index.chunks`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.config import config
from rbtr.domain.models import ChunkKind
from rbtr.index.refs import resolve_read_ref
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration, QueryExtraction
from rbtr.testing import Indexed


@pytest.mark.parametrize(
    "lang",
    sorted(
        lang
        for lang in get_manager().all_language_ids()
        if (registration := get_manager().get_registration(lang)) is not None
        and (registration.extensions or registration.filenames or lang == "plaintext")
    ),
)
def test_empty_source_yields_host_presence(indexed: Indexed, tmp_path: Path, lang: str) -> None:
    """Empty source yields one content-less host-presence chunk, for every
    registered language.

    Records the file's host language so the blob-dedup gate skips an empty
    file on later builds instead of re-parsing it every time.
    """
    registration = get_manager().get_registration(lang)
    assert registration is not None
    if registration.filenames:
        path = min(registration.filenames)
    elif registration.extensions:
        path = f"input{min(registration.extensions)}"
    else:
        path = "input"
    with get_manager().limited_to(registration), indexed({path: ""}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path=path).rows(named=True)
    assert [(c["content"], c["language"]) for c in chunks] == [("", lang)]


def test_chunk_is_unnamed_when_name_capture_missing(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """Chunks get an empty name when the query omits the name capture."""
    query_no_name = "(function_definition) @function\n"
    src = """\
def hello():
    pass
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query_no_name),
    )
    with get_manager().limited_to(reg), indexed({"test.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="test.py").rows(named=True)
    assert len(chunks) >= 1
    assert chunks[0]["name"] == ""


def test_scope_extractor_owns_scope_address(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """A `scope_extractor` overrides the default scope with its own segments."""
    query = "(function_definition name: (identifier) @_fn_name) @function\n"
    src = """\
def hello():
    pass
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query),
    )
    reg.scope_extractor(lambda _resolver, _cap, _node, _caps: ["a", "b"])
    with get_manager().limited_to(reg), indexed({"test.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="test.py").rows(named=True)
    assert len(chunks) == 1
    assert chunks[0]["scope"] == "a::b"


def test_an_enclosing_scope_is_named_by_its_own_capture(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """A scope node with no `name` field takes the name its capture gave it.

    A Markdown `section` has no `name` field; its heading names it.
    """
    query = "(section (atx_heading (inline) @_section_name)) @doc_section\n"
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_markdown",
        extensions=frozenset({".md"}),
        extraction=QueryExtraction(query=query, scope_types=frozenset({"section"})),
    )
    src = """\
# Title

## Install

### Pip

pip.

## Use

Call it.
"""
    with get_manager().limited_to(reg), indexed({"doc.md": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="doc.md").rows(named=True)
    assert [(c["name"], c["scope"]) for c in chunks if c["kind"] == ChunkKind.DOC_SECTION] == [
        ("Title", ""),
        ("Install", "Title"),
        ("Pip", "Title::Install"),
        ("Use", "Title"),
    ]


def test_a_captured_reference_names_its_fragment(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """`path#fragment` names `fragment` in `path`; a fragment alone names it in the file itself."""
    query = """\
(call
  function: (identifier) @_f
  arguments: (argument_list (string (string_content) @_import_module))
  (#eq? @_f "load")) @import
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".html"}),
        extraction=QueryExtraction(query=query),
    )
    src = """\
load("page.html#intro")
load("#usage")
load("plain.html")
"""
    with get_manager().limited_to(reg), indexed({"docs/guide.html": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="docs/guide.html").rows(named=True)
    assert [
        (c["metadata"]["module"], c["metadata"]["names"])
        for c in chunks
        if c["kind"] == ChunkKind.IMPORT
    ] == [("page.html", "intro"), ("guide.html", "usage"), ("plain.html", "")]


def test_a_section_ends_where_its_first_subsection_begins(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """A captured section's chunk holds its own text; each subsection is a chunk of its own."""
    query = "(section (atx_heading (inline) @_section_name)) @doc_section\n"
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_markdown",
        extensions=frozenset({".md"}),
        extraction=QueryExtraction(query=query, scope_types=frozenset({"section"})),
    )
    src = """\
# Title

Intro.

## Install

Run it.
"""
    with get_manager().limited_to(reg), indexed({"doc.md": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="doc.md").rows(named=True)
    assert [(c["name"], c["content"], c["line_start"], c["line_end"]) for c in chunks] == [
        (
            "Title",
            """\
# Title

Intro.""",
            1,
            3,
        ),
        (
            "Install",
            """\
## Install

Run it.""",
            5,
            7,
        ),
    ]


def test_a_long_left_over_node_is_split_into_its_children(
    indexed: Indexed,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node no capture covers, longer than `chunk_lines`, becomes fragments of its children."""
    monkeypatch.setattr(config, "chunk_lines", 4)
    query = "(section (atx_heading (inline) @_section_name)) @doc_section\n"
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_markdown",
        extensions=frozenset({".md"}),
        extraction=QueryExtraction(query=query),
    )
    src = """\
One.

Two.

Three.

Four.
"""
    with get_manager().limited_to(reg), indexed({"doc.md": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="doc.md").rows(named=True)
    assert [(c["content"], c["line_start"]) for c in chunks if c["kind"] == ChunkKind.FRAGMENT] == [
        ("One.", 1),
        ("Two.", 3),
        ("Three.", 5),
        ("Four.", 7),
    ]


def test_strip_directive_strips_a_captured_name(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """`#strip!` removes what its regex matches from the name a capture gives."""
    query = """\
((function_definition name: (identifier) @_fn_name) @function
  (#strip! @_fn_name "^_+"))
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query),
    )
    src = """\
def __hidden():
    pass
"""
    with get_manager().limited_to(reg), indexed({"a.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="a.py").rows(named=True)
    assert [c["name"] for c in chunks if c["kind"] == ChunkKind.FUNCTION] == ["hidden"]


def test_strip_directive_shapes_an_import(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """`#strip!` on `@_import_module` and `@_import_names` shapes the import's module and names."""
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
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query),
    )
    src = 'ref("pkg:guide", "~intro")\n'
    with get_manager().limited_to(reg), indexed({"a.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        rows = index.chunks(at=at, file_path="a.py").rows(named=True)
    [meta] = [c["metadata"] for c in rows if c["kind"] == ChunkKind.IMPORT]
    assert (meta["module"], meta["names"]) == ("guide", "intro")


def test_a_quantified_import_capture_names_one_module_per_line(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """Nodes of a quantified `@_import_module` make one import per line they sit on.

    For a grammar that gives a list's entries no node of their own; each
    line's import is named and spanned by that line, its `#strip!` applied.
    """
    query = """\
((list ((string (string_content) @_import_module) ","?)+) @import
  (#strip! @_import_module "^x:"))
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query),
    )
    src = """\
LINKS = [
    "x:topics/intro",
    "x:api/index",
]
"""
    with get_manager().limited_to(reg), indexed({"a.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        rows = index.chunks(at=at, file_path="a.py").rows(named=True)
    imports = [
        (c["name"], c["metadata"]["module"], c["line_start"], c["line_end"])
        for c in rows
        if c["kind"] == ChunkKind.IMPORT
    ]
    assert imports == [
        ("x:topics/intro", "topics/intro", 2, 2),
        ("x:api/index", "api/index", 3, 3),
    ]


def test_set_directive_names_where_an_import_finds_its_name(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """`#set! import.language_hint` says which language's chunks hold the name, as a module would."""
    query = """\
((call function: (identifier) @_f arguments: (argument_list (string (string_content) @_import_names))) @import
  (#set! import.language_hint "rst")
  (#eq? @_f "ref"))
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query),
    )
    with get_manager().limited_to(reg), indexed({"a.py": 'ref("intro")\n'}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        rows = index.chunks(at=at, file_path="a.py").rows(named=True)
    [meta] = [c["metadata"] for c in rows if c["kind"] == ChunkKind.IMPORT]
    assert (meta["names"], meta["language_hint"]) == ("intro", "rst")


def test_unknown_capture_name_ignored(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """Captures not in CAPTURE_KINDS are silently skipped."""
    query_unknown = """\
(function_definition
  name: (identifier) @_fn_name) @function
(class_definition) @unknown_thing
"""
    src = """\
def f():
    pass

class C:
    pass
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query_unknown),
    )
    with get_manager().limited_to(reg), indexed({"test.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="test.py").rows(named=True)
    kinds = {c["kind"] for c in chunks}
    assert "function" in kinds
    assert "class" not in kinds


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        pytest.param(
            """\
# one
# two
# three
""",
            [
                (
                    1,
                    3,
                    """\
# one
# two
# three""",
                ),
            ],
            id="contiguous_run_is_one_ordered_block",
        ),
        pytest.param(
            """\
# a1
# a2

# b1
# b2
""",
            [
                (
                    1,
                    2,
                    """\
# a1
# a2""",
                ),
                (
                    4,
                    5,
                    """\
# b1
# b2""",
                ),
            ],
            id="blank_line_splits_into_two_source_ordered_blocks",
        ),
        pytest.param(
            """\
# doc
def f():
    pass


# tail
""",
            [(6, 6, "# tail")],
            id="code_splits_run_and_leading_block_folds_into_symbol",
        ),
    ],
)
def test_comment_blocks_group_in_source_order(
    indexed: Indexed, tmp_path: Path, src: str, expected: list[tuple[int, int, str]]
) -> None:
    """Comment runs group into blocks whose lines stay in source order.

    A blank-free run is one block with its lines contiguous and ordered; a
    blank line splits a run into separate blocks emitted in source order; code
    between comments splits them (and a block flush before a symbol folds into
    it rather than standing alone).
    """
    query = """\
(comment) @comment
(function_definition name: (identifier) @_fn_name) @function
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query),
    )
    with get_manager().limited_to(reg), indexed({"t.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="t.py").rows(named=True)
    comment_chunks = [c for c in chunks if c["kind"] == ChunkKind.COMMENT]
    comments = [(c["line_start"], c["line_end"], c["content"]) for c in comment_chunks]
    assert comments == expected
    assert [c["name"] for c in comment_chunks] == [""] * len(comment_chunks), (
        "a comment has no identifier to name it"
    )


def test_trailing_comment_does_not_fold_into_next_symbol(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """A comment trailing code on its line is not the next symbol's doc."""
    query = """\
(comment) @comment
(module (expression_statement (assignment left: (identifier) @_var_name) @variable))
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query),
    )
    src = """\
x = 1  # trailing about x
y = 2
"""
    with get_manager().limited_to(reg), indexed({"t.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="t.py").rows(named=True)
    y = next(c for c in chunks if c["name"] == "y")
    assert "trailing about x" not in y["content"]
    assert any(
        c["kind"] == ChunkKind.COMMENT and "trailing about x" in c["content"] for c in chunks
    )


def test_trailing_comment_not_grouped_with_following_leading_doc(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    """A trailing comment is separate from a real own-line doc for the next symbol."""
    query = """\
(comment) @comment
(function_definition name: (identifier) @_fn_name) @function
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query),
    )
    src = """\
x = 1  # trailing
# real doc
def f():
    pass
"""
    with get_manager().limited_to(reg), indexed({"t.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        rows = index.chunks(at=at, file_path="t.py").rows(named=True)
    f = next(c for c in rows if c["name"] == "f")
    assert "# real doc" in f["content"]
    assert "# trailing" not in f["content"]


@pytest.mark.parametrize(
    ("query", "src", "expected"),
    [
        pytest.param(
            """\
(import_statement) @import
(function_definition name: (identifier) @_fn_name) @function
""",
            """\
import os

setup()
configure()

def f():
    pass

if __name__ == "__main__":
    f()
""",
            [
                (
                    3,
                    4,
                    """\
setup()
configure()""",
                    ("expression_statement",),
                ),
                (
                    9,
                    10,
                    """\
if __name__ == "__main__":
    f()""",
                    ("if_statement",),
                ),
            ],
            id="statements_join_until_a_blank_line_or_a_definition",
        ),
        pytest.param(
            "(import_statement) @import\n",
            """\
if TYPE_CHECKING:
    import os
""",
            [(1, 1, "if TYPE_CHECKING:", ("identifier",))],
            id="a_partly_covered_statement_leaves_only_its_own_lines",
        ),
        pytest.param(
            "(function_definition name: (identifier) @_fn_name) @function\n",
            "x = (\n",
            [(1, 1, "x = (", ("ERROR",))],
            id="a_parse_error_is_a_fragment",
        ),
    ],
)
def test_code_no_pattern_names_becomes_fragments(
    indexed: Indexed,
    tmp_path: Path,
    query: str,
    src: str,
    expected: list[tuple[int, int, str, tuple[str, ...]]],
) -> None:
    """Every line a query leaves reaches a fragment, recorded with the syntax it holds."""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query),
    )
    with get_manager().limited_to(reg), indexed({"t.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="t.py").rows(named=True)
    fragments = [
        (c["line_start"], c["line_end"], c["content"], tuple(c["metadata"]["node_types"]))
        for c in chunks
        if c["kind"] == ChunkKind.FRAGMENT
    ]
    assert fragments == expected
    assert {c["language"] for c in chunks if c["kind"] == ChunkKind.FRAGMENT} <= {"faketest"}


def test_a_fragment_is_scoped_to_the_scopes_around_it(
    indexed: Indexed,
    tmp_path: Path,
) -> None:
    query = "(function_definition name: (identifier) @_fn_name) @function\n"
    src = """\
class Config:
    retries = 3

    def load(self):
        pass
"""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query, scope_types=frozenset({"class_definition"})),
    )
    with get_manager().limited_to(reg), indexed({"t.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="t.py").rows(named=True)
    fragments = [(c["content"], c["scope"]) for c in chunks if c["kind"] == ChunkKind.FRAGMENT]
    assert fragments == [("class Config:", "Config"), ("retries = 3", "Config")]


def test_a_run_of_code_splits_at_the_chunk_size(
    indexed: Indexed, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "chunk_lines", 2)
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query="(comment) @comment\n"),
    )
    src = """\
a()
b()
c()
"""
    with get_manager().limited_to(reg), indexed({"t.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="t.py").rows(named=True)
    assert [(c["line_start"], c["line_end"]) for c in chunks] == [(1, 2), (3, 3)]


def test_a_block_left_to_its_injection_is_not_a_fragment(indexed: Indexed, tmp_path: Path) -> None:
    """An HTML `<script>`'s content belongs to the javascript its injection extracts; a tag on a line of its own is the page's."""
    manager = get_manager()
    html = manager.get_registration("html")
    javascript = manager.get_registration("javascript")
    assert html is not None
    assert javascript is not None
    src = """\
<p>intro</p>
<script>
run();
</script>
"""
    with get_manager().limited_to(html, javascript), indexed({"page.html": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="page.html").rows(named=True)
    fragments = [
        (c["line_start"], c["line_end"], c["content"])
        for c in chunks
        if c["kind"] == ChunkKind.FRAGMENT and c["language"] == "html"
    ]
    assert fragments == [(1, 1, "<p>intro</p>"), (4, 4, "</script>")]


@pytest.mark.parametrize(
    ("query", "src", "expected"),
    [
        pytest.param(
            """\
(import_from_statement
  module_name: (dotted_name) @_import_module
  name: (dotted_name) @_import_names) @import
""",
            "from m import a, b\n",
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
            "from m import a, b as c\n",
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
            "import numpy as np\n",
            [{"module": "numpy", "names": "", "bindings": [{"name": "numpy", "local": "np"}]}],
            id="a-local-alone-binds-the-module",
        ),
        pytest.param(
            """\
(import_statement name: (dotted_name) @_import_module) @import
""",
            "import os, sys\n",
            [
                {"module": "os", "names": "", "bindings": []},
                {"module": "sys", "names": "", "bindings": []},
            ],
            id="two-modules-are-two-imports",
            marks=pytest.mark.xfail(
                strict=True,
                reason="two imports of one statement share its name and lines, so their chunk "
                "ids collide and the index keeps one",
            ),
        ),
        pytest.param(
            """\
((import_from_statement
  module_name: (dotted_name) @_import_module
  (wildcard_import))
  (#set! import.names "*")) @import
""",
            "from m import *\n",
            [{"module": "m", "names": "*", "bindings": []}],
            id="a-pattern-sets-names",
        ),
        pytest.param(
            """\
((import_statement name: (dotted_name) @_import_module)
  (#set! import.prefix "./")) @import
""",
            "import helpers\n",
            [{"module": "./helpers", "names": "", "bindings": []}],
            id="a-pattern-writes-a-prefix-before-its-module",
        ),
    ],
)
def test_a_query_states_an_imports_names_and_bindings(
    indexed: Indexed, tmp_path: Path, query: str, src: str, expected: list[dict]
) -> None:
    """Names, bindings and directives come from captures alone, with no import extractor."""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query=query),
    )
    with get_manager().limited_to(reg), indexed({"t.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        rows = index.chunks(at=at, file_path="t.py").rows(named=True)
    imports = [
        {key: c["metadata"][key] for key in ("module", "names", "bindings")}
        for c in rows
        if c["kind"] == ChunkKind.IMPORT
    ]
    assert imports == expected


def test_a_partly_covered_leaf_is_a_fragment_of_its_own(indexed: Indexed, tmp_path: Path) -> None:
    """A leaf spanning lines, one of them covered, has no children to look into: it is its own fragment."""
    reg = LanguageRegistration(
        id="faketest",
        grammar_module="tree_sitter_python",
        extensions=frozenset({".py"}),
        extraction=QueryExtraction(query="(identifier) @variable\n"),
    )
    src = '''\
x = """a
b
c"""
'''
    with get_manager().limited_to(reg), indexed({"t.py": src}) as index:
        at = resolve_read_ref(index, index.repo(str(tmp_path)), None).at
        chunks = index.chunks(at=at, file_path="t.py").rows(named=True)
    assert [
        (c["line_start"], c["line_end"], tuple(c["metadata"]["node_types"]))
        for c in chunks
        if c["kind"] == ChunkKind.FRAGMENT
    ] == [(1, 3, ("string_content",))]
