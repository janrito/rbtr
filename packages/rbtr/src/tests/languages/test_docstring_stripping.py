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


def _extract_with_doc_types(
    lang: str,
    src: str,
    doc_types: frozenset[str],
    *,
    strip: bool = False,
):
    """Run extraction for *lang* with an explicit leading-comment set.

    Lets the tests exercise the engine-side sibling-walk without
    touching the production plugin config.
    """
    mgr = get_manager()
    grammar = mgr.load_grammar(lang)
    assert grammar is not None, f"grammar for {lang} not installed"
    reg = mgr.get_registration(lang)
    assert reg is not None
    assert reg.query is not None
    ext = next(iter(reg.extensions), ".txt")
    return extract_symbols(
        f"test{ext}",
        "sha1",
        src.encode(),
        grammar,
        reg.query,
        import_extractor=reg.import_extractor,
        scope_types=reg.scope_types,
        doc_comment_node_types=doc_types,
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


# ── Leading-comment attachment (engine sibling walk) ──────────────────


def test_leading_comments_disabled_by_empty_doc_types() -> None:
    """Empty `doc_comment_node_types` → chunks unchanged."""
    src = "// Doc for foo.\nfunc foo() {}\n"
    chunks = _extract_with_doc_types("go", src, frozenset())
    fn = next(c for c in chunks if c.name == "foo")
    assert "Doc for foo" not in fn.content
    assert fn.line_start == 2


def test_leading_comments_attached_when_opted_in() -> None:
    """Opted-in plugin gets the leading comment in chunk content."""
    src = "// Doc for foo.\nfunc foo() {}\n"
    chunks = _extract_with_doc_types("go", src, frozenset({"comment"}))
    fn = next(c for c in chunks if c.name == "foo")
    assert "Doc for foo" in fn.content
    assert fn.line_start == 1
    assert fn.line_end == 2


def test_leading_comments_span_multiple_lines() -> None:
    """Consecutive comments attach as a block."""
    src = "// Line one.\n// Line two.\n// Line three.\nfunc foo() {}\n"
    chunks = _extract_with_doc_types("go", src, frozenset({"comment"}))
    fn = next(c for c in chunks if c.name == "foo")
    assert "Line one" in fn.content
    assert "Line two" in fn.content
    assert "Line three" in fn.content
    assert fn.line_start == 1


def test_leading_comments_stop_at_blank_line() -> None:
    """A blank line between a comment and the symbol breaks attachment."""
    src = "// Not attached.\n\n// Attached.\nfunc foo() {}\n"
    chunks = _extract_with_doc_types("go", src, frozenset({"comment"}))
    fn = next(c for c in chunks if c.name == "foo")
    assert "Attached" in fn.content
    assert "Not attached" not in fn.content
    assert fn.line_start == 3


def test_leading_comments_only_count_if_last_sibling() -> None:
    """Non-comment sibling between comment and symbol breaks attachment."""
    src = (
        "// Header comment.\n"
        "func other() {}\n"  # comment belongs to `other`, not `foo`
        "func foo() {}\n"
    )
    chunks = _extract_with_doc_types("go", src, frozenset({"comment"}))
    foo = next(c for c in chunks if c.name == "foo")
    other = next(c for c in chunks if c.name == "other")
    assert "Header comment" in other.content
    assert "Header comment" not in foo.content


def test_leading_comments_chunk_id_uses_extended_line_start() -> None:
    """Chunk ID hashes the *attached* line_start, not the symbol's own."""
    src = "// Attached.\nfunc foo() {}\n"
    with_doc = _extract_with_doc_types("go", src, frozenset({"comment"}))
    without_doc = _extract_with_doc_types("go", src, frozenset())
    with_id = next(c.id for c in with_doc if c.name == "foo")
    without_id = next(c.id for c in without_doc if c.name == "foo")
    assert with_id != without_id


def test_leading_comments_stripped_with_flag() -> None:
    """`--strip-docstrings` blanks attached comments too."""
    src = "// Doc for foo.\nfunc foo() {}\n"
    chunks = _extract_with_doc_types("go", src, frozenset({"comment"}), strip=True)
    fn = next(c for c in chunks if c.name == "foo")
    assert "Doc for foo" not in fn.content
    assert "func foo()" in fn.content
    # Newlines preserved → two-line content still spans two lines.
    assert fn.line_start == 1
    assert fn.line_end == 2


def test_leading_comments_rust_line_comment_type() -> None:
    """Rust uses `line_comment`, not `comment`."""
    src = "/// Doc.\nfn foo() {}\n"
    chunks = _extract_with_doc_types("rust", src, frozenset({"line_comment", "block_comment"}))
    fn = next(c for c in chunks if c.name == "foo")
    assert "/// Doc." in fn.content
    assert fn.line_start == 1


# ── Plugin defaults (production behaviour) ──────────────────────────


def _plugin_extract(lang: str, src: str, *, strip: bool = False):
    """Extract using the plugin's **production** configuration.

    Unlike `_extract_with_doc_types`, this uses whatever the
    plugin actually registers — so it verifies default-on
    behaviour for exterior-comment languages.
    """
    mgr = get_manager()
    grammar = mgr.load_grammar(lang)
    assert grammar is not None, f"grammar for {lang} not installed"
    reg = mgr.get_registration(lang)
    assert reg is not None
    assert reg.query is not None
    ext = next(iter(reg.extensions), ".txt")
    return extract_symbols(
        f"test{ext}",
        "sha1",
        src.encode(),
        grammar,
        reg.query,
        import_extractor=reg.import_extractor,
        scope_types=reg.scope_types,
        doc_comment_node_types=reg.doc_comment_node_types,
        strip_docstrings=strip,
    )


def test_plugin_defaults_go_attaches_leading_comments() -> None:
    chunks = _plugin_extract("go", "// Foo docs.\nfunc Foo() {}\n")
    fn = next(c for c in chunks if c.name == "Foo")
    assert "Foo docs" in fn.content
    assert fn.line_start == 1


def test_plugin_defaults_rust_attaches_doc_comments() -> None:
    chunks = _plugin_extract("rust", "/// Foo docs.\nfn foo() {}\n")
    fn = next(c for c in chunks if c.name == "foo")
    assert "Foo docs" in fn.content


def test_plugin_defaults_js_attaches_jsdoc() -> None:
    chunks = _plugin_extract("javascript", "/** JSDoc for foo. */\nfunction foo() {}\n")
    fn = next(c for c in chunks if c.name == "foo")
    assert "JSDoc for foo" in fn.content


def test_plugin_defaults_typescript_attaches_jsdoc() -> None:
    chunks = _plugin_extract("typescript", "/** JSDoc for foo. */\nfunction foo() {}\n")
    fn = next(c for c in chunks if c.name == "foo")
    assert "JSDoc for foo" in fn.content


def test_plugin_defaults_java_attaches_javadoc() -> None:
    src = "class A {\n    /** Javadoc for m. */\n    void m() {}\n}\n"
    chunks = _plugin_extract("java", src)
    m = next(c for c in chunks if c.name == "m")
    assert "Javadoc for m" in m.content


def test_plugin_defaults_ruby_attaches_hash_comments() -> None:
    chunks = _plugin_extract("ruby", "# Doc for foo.\ndef foo\nend\n")
    fn = next(c for c in chunks if c.name == "foo")
    assert "Doc for foo" in fn.content


def test_plugin_defaults_c_attaches_leading_comments() -> None:
    chunks = _plugin_extract("c", "/** Doc for foo. */\nint foo(void) { return 0; }\n")
    fn = next(c for c in chunks if c.name == "foo")
    assert "Doc for foo" in fn.content


def test_plugin_defaults_cpp_attaches_leading_comments() -> None:
    chunks = _plugin_extract("cpp", "/** Doc for foo. */\nint foo() { return 0; }\n")
    fn = next(c for c in chunks if c.name == "foo")
    assert "Doc for foo" in fn.content


def test_plugin_defaults_bash_attaches_hash_comments() -> None:
    chunks = _plugin_extract("bash", "# Doc for foo.\nfoo() {\n  echo hi\n}\n")
    fn = next(c for c in chunks if c.name == "foo")
    assert "Doc for foo" in fn.content


def test_plugin_defaults_strip_removes_attached_comments() -> None:
    """End-to-end: with production plugin config, --strip-docstrings
    blanks leading comments across languages.
    """
    src = "// Foo docs.\nfunc Foo() {}\n"
    kept = _plugin_extract("go", src)
    stripped = _plugin_extract("go", src, strip=True)
    foo_kept = next(c for c in kept if c.name == "Foo")
    foo_stripped = next(c for c in stripped if c.name == "Foo")
    assert "Foo docs" in foo_kept.content
    assert "Foo docs" not in foo_stripped.content
    assert foo_kept.line_start == foo_stripped.line_start
    assert foo_kept.line_end == foo_stripped.line_end


def test_plugin_defaults_python_unchanged_no_leading_attachment() -> None:
    """Python deliberately doesn't attach leading `#` comments —
    the interior `@_docstring` capture is the canonical form.
    """
    chunks = _plugin_extract("python", "# leading note\ndef add(a, b):\n    return a + b\n")
    fn = next(c for c in chunks if c.name == "add")
    assert "leading note" not in fn.content
    assert fn.line_start == 2


def test_leading_comments_python_unaffected() -> None:
    """Python doesn't set `doc_comment_node_types`; interior docstring wins."""
    src = '''\
# leading comment
def greet():
    """interior docstring"""
    return 1
'''
    # Explicit empty set to mirror Python's production config.
    chunks = _extract_with_doc_types("python", src, frozenset())
    fn = next(c for c in chunks if c.name == "greet")
    assert "leading comment" not in fn.content
    assert "interior docstring" in fn.content
    assert fn.line_start == 2
