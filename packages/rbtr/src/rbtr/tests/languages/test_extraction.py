"""Engine-level extraction tests.

Per-language extraction lives in each `rbtr-lang-*` package. These tests pin
the language-agnostic engine mechanics: the name/scope resolver fallbacks and
capture-kind filtering in `extract_symbols` (exercised with synthetic
registrations over the Python grammar), comment grouping/folding, and the
content-less host-presence chunk emitted for every registered language.
"""

from __future__ import annotations

import pytest

from rbtr.git import FileEntry
from rbtr.index.models import ChunkKind
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


def test_anonymous_chunk_when_name_capture_missing() -> None:
    """Chunks get name='<anonymous>' when the query omits the name capture."""
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
    assert chunks[0].name == "<anonymous>"


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
    comments = [
        (c.line_start, c.line_end, c.content) for c in chunks if c.kind == ChunkKind.COMMENT
    ]
    assert comments == expected


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
