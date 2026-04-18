"""Tests for the optional docstring-stripping behaviour.

When `extract_symbols` is called with `strip_docstrings=True`,
bytes covered by the query's `@_docstring` sub-capture are
replaced with whitespace in the chunk content while preserving
line count so `line_start` / `line_end` remain valid.
"""

from __future__ import annotations

from tree_sitter import QueryCursor

from rbtr.index.treesitter import _DOCSTRING_CAPTURE, _get_query, extract_symbols
from rbtr.languages import get_manager


def _py_extract(src: str, *, strip: bool) -> list:
    """Run Python extraction with or without docstring stripping."""
    mgr = get_manager()
    grammar = mgr.load_grammar("python")
    assert grammar is not None
    reg = mgr.get_registration("python")
    assert reg is not None
    assert reg.query is not None
    return extract_symbols(
        "test.py",
        "sha1",
        src.encode(),
        grammar,
        reg.query,
        import_extractor=reg.import_extractor,
        scope_types=reg.scope_types,
        strip_docstrings=strip,
    )


def test_strip_removes_function_docstring_text() -> None:
    src = '''\
def greet(name):
    """Say hello to *name*."""
    return f"hi {name}"
'''
    chunks = _py_extract(src, strip=True)
    fn = next(c for c in chunks if c.name == "greet")
    assert "Say hello" not in fn.content
    assert "return f" in fn.content


def test_no_strip_keeps_docstring() -> None:
    src = '''\
def greet(name):
    """Say hello to *name*."""
    return f"hi {name}"
'''
    chunks = _py_extract(src, strip=False)
    fn = next(c for c in chunks if c.name == "greet")
    assert "Say hello" in fn.content


def test_strip_preserves_line_numbers() -> None:
    src = '''\
def greet(name):
    """Line one.

    Line three.
    """
    return name
'''
    chunks = _py_extract(src, strip=True)
    fn = next(c for c in chunks if c.name == "greet")
    # Original spans lines 1..6.  Stripped content must still span
    # the same number of lines so line_end is unchanged.
    assert fn.line_start == 1
    assert fn.line_end == 6
    assert fn.content.count("\n") == src.rstrip("\n").count("\n")
    assert "Line one" not in fn.content
    assert "Line three" not in fn.content
    assert "return name" in fn.content


def test_strip_class_docstring() -> None:
    src = '''\
class Greeter:
    """A polite greeter."""

    def greet(self):
        return "hi"
'''
    chunks = _py_extract(src, strip=True)
    cls = next(c for c in chunks if c.name == "Greeter")
    assert "polite greeter" not in cls.content
    # Method body still present (the class chunk contains the full class).
    assert "def greet" in cls.content


def test_strip_leaves_non_docstring_strings_alone() -> None:
    src = """\
def echo():
    x = "this is not a docstring"
    return x
"""
    chunks = _py_extract(src, strip=True)
    fn = next(c for c in chunks if c.name == "echo")
    assert "this is not a docstring" in fn.content


def test_strip_noop_when_no_docstring() -> None:
    src = """\
def add(a, b):
    return a + b
"""
    before = _py_extract(src, strip=False)
    after = _py_extract(src, strip=True)
    assert before[0].content == after[0].content


def test_query_yields_docstring_capture_for_documented_function() -> None:
    """The Python query exposes a `@_docstring` capture.

    Guards the capture convention `extract_symbols` relies on.
    """
    mgr = get_manager()
    grammar = mgr.load_grammar("python")
    assert grammar is not None
    reg = mgr.get_registration("python")
    assert reg is not None
    assert reg.query is not None

    src = b'''\
def greet():
    """hello"""
    return 1
'''
    from tree_sitter import Parser

    tree = Parser(grammar).parse(src)
    query = _get_query(grammar, reg.query)
    matches = QueryCursor(query).matches(tree.root_node)

    doc_captures = [
        node
        for _, caps in matches
        for key, nodes in caps.items()
        if key == _DOCSTRING_CAPTURE
        for node in nodes
    ]
    assert len(doc_captures) == 1
    assert doc_captures[0].text == b'"""hello"""'


def test_query_no_docstring_capture_when_absent() -> None:
    """Functions without docstrings don't yield a `@_docstring` capture."""
    mgr = get_manager()
    grammar = mgr.load_grammar("python")
    assert grammar is not None
    reg = mgr.get_registration("python")
    assert reg is not None
    assert reg.query is not None

    src = b"def add(a, b):\n    return a + b\n"
    from tree_sitter import Parser

    tree = Parser(grammar).parse(src)
    query = _get_query(grammar, reg.query)
    matches = QueryCursor(query).matches(tree.root_node)

    doc_captures = [
        node
        for _, caps in matches
        for key, nodes in caps.items()
        if key == _DOCSTRING_CAPTURE
        for node in nodes
    ]
    assert doc_captures == []
