"""Engine-level extraction tests.

Per-language extraction lives in each `rbtr-lang-*` package. These tests pin
the language-agnostic engine mechanics: the name/scope resolver fallbacks and
capture-kind filtering in `extract_symbols` (exercised with synthetic
registrations over the Python grammar), comment grouping/folding, and the
content-less host-presence chunk emitted for every registered language.
"""

from __future__ import annotations

import pytest

from rbtr.config import config
from rbtr.domain.models import ChunkKind, Fragment, Section
from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration, QueryExtraction
from rbtr.languages.treesitter import extract_symbols


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


def test_anchor_extractor_gives_a_section_its_anchors() -> None:
    """An `anchor_extractor` sets the anchors of the sections a query captures."""
    grammar = get_manager().grammar("python")
    assert grammar is not None
    query = "(class_definition name: (identifier) @_section_name) @doc_section\n"
    src = b"""\
class Intro:
    pass
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    reg.anchor_extractor(lambda _resolver, _node, _captures: ("intro", "the-intro"))
    chunks = list(extract_symbols(reg, "test.py", "sha1", src, grammar))
    assert [c.role.meta.anchors for c in chunks if isinstance(c.role, Section)] == [
        ("intro", "the-intro")
    ]


def test_unknown_capture_name_ignored() -> None:
    """Captures not in _CAPTURE_KINDS are silently skipped."""
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
    """An HTML `<script>` belongs to the javascript its injection extracts."""
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
    assert fragments == [(1, 1, "<p>intro</p>")]
