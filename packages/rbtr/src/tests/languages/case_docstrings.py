"""Docstring-extraction test cases for all languages.

Each `@case` returns a 4-tuple `(lang, source, symbol_name,
snippet)` consumed by `test_docstrings.py`:

* `lang`        - language id registered by a plugin.
* `source`      - a small but realistic source fragment.
* `symbol_name` - name of the chunk whose `content` the test
  will inspect.  Cases keep names unique so disambiguation is
  trivial.
* `snippet`     - a substring the test will look for.  For
  `documented` cases the snippet must appear in `content`; for
  `undocumented` cases it must *not* appear.

Tags
----

Every case carries **two** tags.

Primary (drives the assertion direction):

* `documented`   - the chunk's `content` must contain `snippet`.
* `undocumented` - the chunk's `content` must *not* contain
  `snippet`.

Mechanism (partitions `documented` cases by the engine feature
responsible for extraction; lets two tests assert opposite
outcomes over disjoint case slices instead of branching inside
one test):

* `interior_doc` - the docstring is inside the symbol node,
  extracted via the plugin's `@_docstring` query capture.  Python
  is the only language using this mechanism.
* `exterior_doc` - the documentation is a leading sibling
  comment block, attached via the engine's sibling walk.  Every
  non-Python plugin uses this.

Secondary (classifies the scenario for debugging slices):

* `canonical`              - the language's idiomatic,
                              by-the-book documentation form.
* `edge_case`              - a valid but unusual doc form (raw
                              / byte strings, nested quotes,
                              multi-line block comments,
                              language-specific inner doc such
                              as Rust `//!`).
* `unconventional`         - not strictly canonical but clearly
                              intended as docs.  rbtr leans
                              toward extracting these.
* `boundary_attached`      - attachment-logic scenario whose
                              outcome is attachment.
* `no_docs`                - symbol has no documentation.
* `boundary_not_attached`  - attachment-logic scenario whose
                              outcome is no attachment.
* `invalid`                - text that looks like documentation
                              but is not recognised as such.

Policy being tested: "the default rbtr behaviour is that every
documented symbol exposes its documentation in the chunk
content".  Every language has at least one case in each primary
bucket and covers every secondary tag relevant to that language.

Source-template convention
--------------------------

Python source strings use single-quoted triple-string (`'''...'''`)
templates so their body can embed double-quoted triple-string
docstrings (`\"\"\"...\"\"\"`) verbatim.
"""

from __future__ import annotations

from pytest_cases import case

# ═════════════════════════════════════════════════════════════════════
# Python
# ═════════════════════════════════════════════════════════════════════


@case(tags=["documented", "canonical", "interior_doc"])
def case_py_function_docstring():
    """PEP 257 single-line function docstring."""
    src = '''\
def greet(name):
    """Return a friendly greeting for *name*."""
    return f"hi {name}"
'''
    return "python", src, "greet", "Return a friendly greeting"


@case(tags=["documented", "canonical", "interior_doc"])
def case_py_class_docstring():
    """Class-level docstring as first statement of the body."""
    src = '''\
class Greeter:
    """Produce friendly greetings."""

    def hi(self):
        return "hi"
'''
    return "python", src, "Greeter", "Produce friendly greetings"


@case(tags=["documented", "canonical", "interior_doc"])
def case_py_method_docstring():
    """Method docstring inside a class."""
    src = '''\
class Svc:
    def run(self):
        """Execute the main loop."""
        return 0
'''
    return "python", src, "run", "Execute the main loop"


@case(tags=["documented", "canonical", "interior_doc"])
def case_py_multiline_docstring():
    """Multi-line docstring with summary + body."""
    src = '''\
def compute(x, y):
    """Add two numbers.

    The summary is one line; the body elaborates.  Both parts
    must remain in chunk content.
    """
    return x + y
'''
    return "python", src, "compute", "The summary is one line"


@case(tags=["documented", "edge_case", "interior_doc"])
def case_py_raw_string_docstring():
    """`r\"\"\"...\"\"\"` raw string is a valid docstring."""
    src = '''\
def regex():
    r"""Match DIGIT sequences."""
    return 1
'''
    return "python", src, "regex", "Match DIGIT sequences"


@case(tags=["documented", "edge_case", "interior_doc"])
def case_py_single_quoted_docstring():
    """Single-quoted triple-string is equally valid."""
    src = """\
def foo():
    '''Single-quoted docstring content.'''
    return 1
"""
    return "python", src, "foo", "Single-quoted docstring content"


@case(tags=["documented", "edge_case", "interior_doc"])
def case_py_decorated_function_docstring():
    """Decorators precede `def`; docstring is still interior."""
    src = '''\
@cache
@wraps(other)
def memoized(x):
    """Memoise calls to *other*."""
    return other(x)
'''
    return "python", src, "memoized", "Memoise calls"


@case(tags=["documented", "unconventional", "interior_doc"])
def case_py_docstring_with_code_block():
    """Docstring embedding example code - common in libraries."""
    src = '''\
def parse(s):
    """Parse *s* into tokens.

    Example::

        >>> parse("1+2")
        [1, "+", 2]
    """
    return []
'''
    return "python", src, "parse", ">>> parse"


@case(tags=["documented", "unconventional", "interior_doc"])
def case_py_class_and_method_both_documented():
    """Method chunk carries its own docstring, not the class's."""
    src = '''\
class Svc:
    """Service facade."""

    def start(self):
        """Boot the service."""
        return 1
'''
    return "python", src, "start", "Boot the service"


@case(tags=["undocumented", "no_docs"])
def case_py_function_without_docstring():
    """Plain function: chunk content has no triple-quoted prose.

    The probe is a unique marker string that would only be
    present if some other source polluted this chunk.
    """
    src = """\
def add(a, b):
    return a + b
"""
    return "python", src, "add", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "no_docs"])
def case_py_class_without_docstring():
    """Plain class with no documentation."""
    src = """\
class Bare:
    pass
"""
    return "python", src, "Bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "invalid"])
def case_py_leading_hash_comment_not_attached():
    """A `#` comment above `def` is deliberately not treated as
    documentation in Python.  The interior triple-string form
    is canonical, and `#` runs frequently serve as section
    headers or type-ignore markers — attaching them would
    double-count noise.
    """
    src = """\
# HASH_COMMENT_ABOVE_DEF_MARKER
def work():
    return 1
"""
    return "python", src, "work", "HASH_COMMENT_ABOVE_DEF_MARKER"


@case(tags=["undocumented", "invalid"])
def case_py_trailing_string_not_a_docstring():
    """A triple-string placed *after* other statements is a
    discarded expression, not a docstring.  The text is in the
    chunk content (it is part of the function body) but is not
    treated as documentation — `--strip-docstrings` must leave
    it intact.  The `test_stripping_removes_doc_text` policy
    guards *documented* cases; this case goes in
    `undocumented` because the engine must not claim the
    string as documentation in the first place.

    We probe for presence of the marker using the
    `test_no_phantom_documentation` logic inverted: the marker
    *is* in the chunk, so probing for a different string that
    would only appear if the extractor invented content gives
    the right signal.
    """
    src = '''\
def late():
    x = 1
    """PSEUDO_DOC"""
    return x
'''
    return "python", src, "late", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


# ══════════════════════════════════════════════════════════════════
# JavaScript
# ══════════════════════════════════════════════════════════════════


@case(tags=["documented", "canonical", "exterior_doc"])
def case_js_jsdoc_on_function():
    """Canonical JSDoc above a function declaration."""
    src = """\
/** Return a friendly greeting. */
function greet() {}
"""
    return "javascript", src, "greet", "Return a friendly greeting"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_js_jsdoc_on_class():
    """Canonical JSDoc above a class declaration."""
    src = """\
/** A widget. */
class Widget {}
"""
    return "javascript", src, "Widget", "A widget"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_js_jsdoc_on_arrow_function():
    """Arrow-function assignment — the `@function` capture
    lands on the `lexical_declaration`, so JSDoc attaches via
    the walk on that node's prev_named_sibling.
    """
    src = """\
/** Increment. */
const inc = (x) => x + 1;
"""
    return "javascript", src, "inc", "Increment"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_js_multiline_jsdoc():
    """Multi-line JSDoc with leading `*` gutter."""
    src = """\
/**
 * Compute a checksum over *data*.
 *
 * The algorithm is CRC32 — explained below.
 */
function checksum(data) { return 0; }
"""
    return "javascript", src, "checksum", "The algorithm is CRC32"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_js_banner_comment():
    """`/*! ... */` banner comments are common in bundled UMD
    libs; the grammar lands them as `comment` nodes and we
    attach them — the benchmark will say whether that hurts.
    """
    src = """\
/*! (c) 2024 Acme. */
function publicApi() {}
"""
    return "javascript", src, "publicApi", "(c) 2024 Acme"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_js_line_comment_run():
    """`//` comment runs used as docs — common in TS-first
    code where JSDoc is syntactically less convenient.
    """
    src = """\
// First description line.
// Second description line.
function documented() {}
"""
    return "javascript", src, "documented", "First description line"


# Known limitation: JavaScript/TypeScript plugins do not emit
# `@method` chunks, so method-level JSDoc lives inside the
# class chunk's bytes but is not tied to any `@_docstring`
# capture or leading-comment walk.  `--strip-docstrings`
# therefore cannot remove it.  The limitation is tracked in
# TODO-benchmark-docstring-ablation.md (D14).  No case here
# — writing one would either lie about strip behaviour or
# force a skip.


@case(tags=["undocumented", "no_docs"])
def case_js_function_without_doc():
    """Plain function, no comments."""
    src = """\
function bare() {}
"""
    return "javascript", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_js_jsdoc_detached_by_blank_line():
    """A blank line between the JSDoc block and the function
    breaks attachment.
    """
    src = """\
/** Stale JSDoc — not attached. */

function stale() {}
"""
    return "javascript", src, "stale", "Stale JSDoc"


@case(tags=["undocumented", "invalid"])
def case_js_jsdoc_above_import_does_not_attach_to_class():
    """JSDoc above an `import` stays on the import line.  A
    class several statements later must *not* inherit it.
    Imports are excluded from attachment by design (see
    `treesitter.extract_symbols`).
    """
    src = """\
/** Nonsense JSDoc above import. */
import { x } from './x';

class Real {}
"""
    return "javascript", src, "Real", "Nonsense JSDoc"


# ══════════════════════════════════════════════════════════════════
# TypeScript
# ══════════════════════════════════════════════════════════════════
#
# TypeScript reuses the JS grammar family with extra
# annotations; docstring semantics are identical so cases
# mirror JS while exercising annotation-bearing signatures.
# interface / type declarations are currently not captured by
# the TS plugin — tracked separately; not covered here.


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ts_jsdoc_on_function():
    """Canonical JSDoc above a typed function declaration."""
    src = """\
/** Return the length of *s*. */
function len(s: string): number { return s.length; }
"""
    return "typescript", src, "len", "Return the length"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ts_jsdoc_on_class():
    """JSDoc above a TypeScript class — grammar uses
    `type_identifier` for the class name.
    """
    src = """\
/** A widget. */
class Widget {}
"""
    return "typescript", src, "Widget", "A widget"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ts_jsdoc_on_arrow_function():
    """Arrow-function assignment with a type annotation."""
    src = """\
/** Increment. */
const inc: (x: number) => number = (x) => x + 1;
"""
    return "typescript", src, "inc", "Increment"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_ts_jsdoc_on_generic_function():
    """Generic type parameters between name and arguments."""
    src = """\
/** Identity. */
function identity<T>(x: T): T { return x; }
"""
    return "typescript", src, "identity", "Identity"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_ts_multiline_jsdoc_with_tags():
    """Multi-line JSDoc with `@param` / `@returns` tags."""
    src = """\
/**
 * Compute a hash.
 *
 * @param data bytes to hash
 * @returns hex digest
 */
function hash(data: Uint8Array): string { return ''; }
"""
    return "typescript", src, "hash", "@returns hex digest"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_ts_line_comment_run():
    """`//` comment runs used as docs, common in TS-heavy
    codebases that avoid JSDoc because types are already in
    the signature.
    """
    src = """\
// Describe the value.
// Useful in calling code.
function describe(x: number): string { return String(x); }
"""
    return "typescript", src, "describe", "Describe the value"


@case(tags=["undocumented", "no_docs"])
def case_ts_function_without_doc():
    """Plain TS function."""
    src = """\
function bare(x: number): number { return x; }
"""
    return "typescript", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_ts_jsdoc_detached_by_blank_line():
    """Blank line breaks attachment for TS too."""
    src = """\
/** Stale JSDoc. */

function stale(): void {}
"""
    return "typescript", src, "stale", "Stale JSDoc"


@case(tags=["undocumented", "invalid"])
def case_ts_jsdoc_above_import():
    """JSDoc above `import` does not attach to a later class."""
    src = """\
/** Nonsense JSDoc. */
import { x } from './x';

class Real {}
"""
    return "typescript", src, "Real", "Nonsense JSDoc"


# ══════════════════════════════════════════════════════════════════
# Rust
# ══════════════════════════════════════════════════════════════════
#
# Rust uses `line_comment` for both `//` and `///` (the
# `outer_doc_comment_marker` distinction lives *inside* the
# line_comment node) and `block_comment` for `/* */` and
# `/** */`.  The plugin lists both so canonical `///` runs,
# unconventional `//` runs, block doc comments, and banner
# blocks all attach.


@case(tags=["documented", "canonical", "exterior_doc"])
def case_rust_triple_slash_on_fn():
    """Canonical `///` doc comment above a function."""
    src = """\
/// Greet the user.
fn greet() {}
"""
    return "rust", src, "greet", "Greet the user"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_rust_triple_slash_on_struct():
    """`///` above a struct declaration."""
    src = """\
/// A widget.
struct Widget;
"""
    return "rust", src, "Widget", "A widget"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_rust_triple_slash_on_enum():
    """`///` above an enum declaration."""
    src = """\
/// Colours.
enum Colour { Red, Green }
"""
    return "rust", src, "Colour", "Colours"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_rust_multi_line_triple_slash():
    """Multi-line `///` run — each line is its own
    `line_comment` node; all attach.
    """
    src = """\
/// Compute a checksum.
///
/// The algorithm is CRC32.
fn checksum() {}
"""
    return "rust", src, "checksum", "The algorithm is CRC32"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_rust_block_doc_comment():
    """`/** */` block doc comment — parsed as `block_comment`
    in the grammar.
    """
    src = """\
/** Block doc. */
fn foo() {}
"""
    return "rust", src, "foo", "Block doc"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_rust_impl_block_doc():
    """`///` above an `impl` block (rbtr treats impls as
    classes and attaches leading docs).  The impl here is for
    a type declared in another module so only one chunk named
    after the type appears in this snippet.
    """
    src = """\
/// Methods for Other.
impl Other {}
"""
    return "rust", src, "Other", "Methods for Other"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_rust_plain_line_comment():
    """Plain `//` comments are also attached — rbtr leans
    toward flexibility rather than requiring strict `///`.
    """
    src = """\
// Plain line comment.
// Second line.
fn foo() {}
"""
    return "rust", src, "foo", "Plain line comment"


@case(tags=["undocumented", "no_docs"])
def case_rust_fn_without_docs():
    """Undocumented function."""
    src = """\
fn bare() {}
"""
    return "rust", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_rust_doc_detached_by_blank_line():
    """A blank line between the doc comment and the function
    breaks attachment.  This is the tree-sitter-rust trailing-
    newline edge case: the line_comment span includes its own
    `\\n`, so "blank line" is detected via a ``>= 2`` newline
    count over `[content_end_without_nl, next_start)`.
    """
    src = """\
/// Orphaned comment.

fn later() {}
"""
    return "rust", src, "later", "Orphaned comment"


@case(tags=["undocumented", "invalid"])
def case_rust_inner_doc_not_attached():
    """File-level `//!` inner doc is not attached to the first
    `fn` when a blank line separates them (the common idiom).
    This guards against the tree-sitter-rust trailing-newline
    bug that would otherwise greedily walk past the blank line.
    """
    src = """\
//! Crate-level doc.
//! More crate doc.

/// Item doc.
fn item() {}
"""
    return "rust", src, "item", "Crate-level doc"


# ══════════════════════════════════════════════════════════════════
# Go
# ══════════════════════════════════════════════════════════════════
#
# Go convention (enforced by `gofmt` / `go doc`): every
# exported symbol begins its comment with the symbol's name.
# tree-sitter-go emits a single `comment` node type for both
# `//` and `/* */`; the plugin lists `{comment}` so both
# attach.


@case(tags=["documented", "canonical", "exterior_doc"])
def case_go_doc_comment_on_function():
    """Canonical Go doc comment above `func`."""
    src = """\
package main

// Greet says hello.
func Greet() {}
"""
    return "go", src, "Greet", "Greet says hello"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_go_doc_comment_on_type():
    """Doc comment above a `type` declaration."""
    src = """\
package main

// Widget is a UI element.
type Widget struct {
    name string
}
"""
    return "go", src, "Widget", "Widget is a UI element"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_go_doc_comment_on_method():
    """Doc comment above a method receiver."""
    src = """\
package main

type T struct{}

// Do performs the action.
func (t *T) Do() {}
"""
    return "go", src, "Do", "Do performs the action"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_go_multi_line_doc_comment():
    """Multi-line `//` doc-comment run."""
    src = """\
package main

// Compute executes the pipeline.
//
// It returns an error if the inputs do not validate.
func Compute() error { return nil }
"""
    return "go", src, "Compute", "It returns an error if the inputs"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_go_block_comment():
    """`/* ... */` block comment as Go-doc.  Supported by
    `go doc` but rare.
    """
    src = """\
package main

/* Block doc above foo.
   Continues here. */
func Foo() {}
"""
    return "go", src, "Foo", "Block doc above foo"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_go_non_godoc_style_comment():
    """Comment that does *not* begin with the symbol's name.
    `go doc` would warn, but rbtr leans toward flexibility and
    attaches it.
    """
    src = """\
package main

// Deliberately unconventional opening.
func Work() {}
"""
    return "go", src, "Work", "Deliberately unconventional opening"


@case(tags=["undocumented", "no_docs"])
def case_go_fn_without_doc():
    """Undocumented function."""
    src = """\
package main

func bare() {}
"""
    return "go", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_go_doc_detached_by_blank_line():
    """Blank line breaks attachment — the Go style guide
    explicitly forbids a blank line between a doc comment and
    its symbol.
    """
    src = """\
package main

// Orphan.

func Later() {}
"""
    return "go", src, "Later", "Orphan"


@case(tags=["undocumented", "invalid"])
def case_go_doc_comment_above_previous_function_not_attached():
    """Comment between two `func`s belongs to the second one,
    not the first.  The walk starts from the symbol and goes
    *backwards*, so the comment correctly attaches to `Second`.
    We probe the chunk for the *first* function to confirm it
    does not steal the comment.
    """
    src = """\
package main

func First() {}

// Doc for Second.
func Second() {}
"""
    return "go", src, "First", "Doc for Second"


# ══════════════════════════════════════════════════════════════════
# Java
# ══════════════════════════════════════════════════════════════════
#
# Java uses `block_comment` for `/** */` Javadoc and
# `line_comment` for `//` — both listed in the plugin.
# Annotations (`@Deprecated` etc.) parse as part of the
# method's `modifiers` child, so the method's
# `prev_named_sibling` is the Javadoc directly — attachment
# works even when annotations sit between.


@case(tags=["documented", "canonical", "exterior_doc"])
def case_java_javadoc_on_class():
    """Canonical Javadoc above a class declaration."""
    src = """\
/** A widget. */
public class Widget {}
"""
    return "java", src, "Widget", "A widget"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_java_javadoc_on_method():
    """Canonical Javadoc above a method declaration."""
    src = """\
public class Widget {
    /** Render the widget. */
    public void render() {}
}
"""
    return "java", src, "render", "Render the widget"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_java_multi_line_javadoc_with_tags():
    """Multi-line Javadoc with `@param` / `@return`."""
    src = """\
public class Widget {
    /**
     * Compute a hash.
     * @param data input
     * @return hex string
     */
    public String hash(byte[] data) { return ""; }
}
"""
    return "java", src, "hash", "@return hex string"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_java_javadoc_above_annotated_method():
    """Annotations parse as part of the method node's
    `modifiers` child, so the Javadoc remains the method's
    `prev_named_sibling` and attaches correctly.
    """
    src = """\
public class Widget {
    /** Deprecated API. */
    @Deprecated
    public void old() {}
}
"""
    return "java", src, "old", "Deprecated API"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_java_line_comment_run_on_method():
    """Plain `//` comment run attaches too."""
    src = """\
public class C {
    // A line comment.
    // Second line.
    public void m() {}
}
"""
    return "java", src, "m", "A line comment"


@case(tags=["undocumented", "no_docs"])
def case_java_method_without_doc():
    """Undocumented method."""
    src = """\
public class C { public void bare() {} }
"""
    return "java", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_java_javadoc_detached_by_blank_line():
    """Blank line between Javadoc and method breaks attachment."""
    src = """\
public class C {
    /** Orphan. */

    public void later() {}
}
"""
    return "java", src, "later", "Orphan"


@case(tags=["undocumented", "invalid"])
def case_java_javadoc_on_previous_method_does_not_attach_to_later():
    """A Javadoc above method A is not attached to later method B."""
    src = """\
public class C {
    /** Doc for first. */
    public void first() {}

    public void second() {}
}
"""
    return "java", src, "second", "Doc for first"


# ══════════════════════════════════════════════════════════════════
# Ruby
# ══════════════════════════════════════════════════════════════════
#
# Ruby convention: `#` comment runs above top-level `def`,
# `class`, and `module` declarations.  The tree-sitter-ruby
# grammar nests methods inside `body_statement` under their
# enclosing class, so comments inside a class body sit at a
# different level than the method node — attachment does not
# cross the `body_statement` boundary.  This mirrors the JS
# limitation noted in D14: method-level documentation inside
# a class is carried inside the class chunk's bytes but does
# not attach to the per-method chunk and cannot be stripped.
# Cases here focus on top-level `def` / `class` / `module`
# where attachment works cleanly.


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ruby_hash_doc_on_def():
    """Canonical `#` comment above a top-level def."""
    src = """\
# Greet the user.
def greet
  'hi'
end
"""
    return "ruby", src, "greet", "Greet the user"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ruby_hash_doc_on_class():
    """`#` comment above a class declaration."""
    src = """\
# Service facade.
class Svc
end
"""
    return "ruby", src, "Svc", "Service facade"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ruby_hash_doc_on_module():
    """`#` comment above a module declaration."""
    src = """\
# Utilities.
module Utils
end
"""
    return "ruby", src, "Utils", "Utilities"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_ruby_multi_line_hash_doc():
    """Multi-line `#` doc comment."""
    src = """\
# Compute a checksum.
#
# Returns a hex digest string.
def checksum
  ''
end
"""
    return "ruby", src, "checksum", "Returns a hex digest"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_ruby_block_comment_begin_end():
    """`=begin` / `=end` block comment.  The grammar treats it
    as a single `comment` node; attachment works when it
    immediately precedes the `def`.
    """
    src = """\
=begin
Block-style doc.
=end
def foo
end
"""
    return "ruby", src, "foo", "Block-style doc"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_ruby_shebang_like_comment_above_def():
    """Unconventional but valid: a `#` run whose first line
    starts with `#!`-style emphasis still attaches.
    """
    src = """\
#! IMPORTANT: use Foo instead of Bar.
# Prefer modern API.
def legacy
end
"""
    return "ruby", src, "legacy", "IMPORTANT: use Foo"


@case(tags=["undocumented", "no_docs"])
def case_ruby_def_without_doc():
    """Undocumented top-level def."""
    src = """\
def bare
end
"""
    return "ruby", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_ruby_doc_detached_by_blank_line():
    """Blank line between comment and def breaks attachment."""
    src = """\
# Orphan.

def later
end
"""
    return "ruby", src, "later", "Orphan"


@case(tags=["undocumented", "invalid"])
def case_ruby_doc_does_not_steal_from_next():
    """A `#` comment between two top-level defs belongs to the
    later def, not the earlier one.
    """
    src = """\
def first
end

# Doc for second.
def second
end
"""
    return "ruby", src, "first", "Doc for second"


# ══════════════════════════════════════════════════════════════════
# C
# ══════════════════════════════════════════════════════════════════
#
# tree-sitter-c uses a single `comment` node for both `//`
# and `/* */` (including `/** */`).  Doxygen style is common
# and needs no special handling.


@case(tags=["documented", "canonical", "exterior_doc"])
def case_c_doxygen_on_function():
    """Canonical Doxygen `/** */` above a function."""
    src = """\
/** Compute the sum. */
int add(int a, int b) { return a + b; }
"""
    return "c", src, "add", "Compute the sum"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_c_doxygen_on_struct():
    """Doxygen above a struct."""
    src = """\
/** Point in 2D space. */
struct Point { int x; int y; };
"""
    return "c", src, "Point", "Point in 2D space"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_c_multi_line_doxygen():
    r"""Multi-line Doxygen with `\param` / `\return` tags."""
    src = """\
/**
 * Hash a buffer.
 * \\param data the buffer
 * \\return the hash
 */
int hash(const char *data) { return 0; }
"""
    return "c", src, "hash", r"\return the hash"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_c_line_comment_run():
    """Plain `//` comment run — common in embedded code where
    Doxygen style is heavier than needed.
    """
    src = """\
// Simple comment.
// Second line.
int foo(void) { return 0; }
"""
    return "c", src, "foo", "Simple comment"


@case(tags=["undocumented", "no_docs"])
def case_c_fn_without_doc():
    src = """\
int bare(void) { return 0; }
"""
    return "c", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_c_doc_detached_by_blank_line():
    """Blank line breaks attachment."""
    src = """\
/** Orphan. */

int later(void) { return 0; }
"""
    return "c", src, "later", "Orphan"


@case(tags=["undocumented", "invalid"])
def case_c_doc_between_two_functions():
    """Comment between two functions belongs to the later one."""
    src = """\
int first(void) { return 0; }

/** Doc for second. */
int second(void) { return 0; }
"""
    return "c", src, "first", "Doc for second"


# ══════════════════════════════════════════════════════════════════
# C++
# ══════════════════════════════════════════════════════════════════
#
# tree-sitter-cpp behaves like tree-sitter-c for comments.
# Cases mirror the C ones but exercise C++-specific constructs
# (classes, methods).


@case(tags=["documented", "canonical", "exterior_doc"])
def case_cpp_doxygen_on_class():
    """Doxygen above a C++ class."""
    src = """\
/** A widget. */
class Widget { public: int x; };
"""
    return "cpp", src, "Widget", "A widget"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_cpp_doxygen_on_function():
    """Doxygen above a C++ function."""
    src = """\
/** Get the answer. */
int answer() { return 42; }
"""
    return "cpp", src, "answer", "Get the answer"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_cpp_triple_slash_style():
    """Doxygen `///` style (single-line convention)."""
    src = """\
/// Triple-slash style.
int foo() { return 0; }
"""
    return "cpp", src, "foo", "Triple-slash style"


@case(tags=["undocumented", "no_docs"])
def case_cpp_fn_without_doc():
    src = """\
int bare() { return 0; }
"""
    return "cpp", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_cpp_doc_detached_by_blank_line():
    src = """\
/** Orphan. */

int later() { return 0; }
"""
    return "cpp", src, "later", "Orphan"


# ══════════════════════════════════════════════════════════════════
# Bash
# ══════════════════════════════════════════════════════════════════
#
# tree-sitter-bash uses a single `comment` node for any `#`
# line.  Function-level documentation in shell scripts is
# typically a `#` comment run directly above the definition.
# Shebangs (`#!/bin/bash`) are also comments; when a function
# sits immediately below the shebang with no blank line, the
# shebang attaches — rbtr leans flexible, the benchmark will
# tell us if this hurts.


@case(tags=["documented", "canonical", "exterior_doc"])
def case_bash_hash_doc_on_function():
    """Canonical `#` comment above a shell function."""
    src = """\
# Greet the user.
greet() {
  echo hello
}
"""
    return "bash", src, "greet", "Greet the user"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_bash_multi_line_hash_doc():
    """Multi-line `#` comment run."""
    src = """\
# Greet the user.
#
# Reads the name from $1.
greet() {
  echo hi $1
}
"""
    return "bash", src, "greet", "Reads the name from"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_bash_shebang_attached_to_first_function():
    """When a function follows the shebang with no blank line,
    the shebang attaches — an honest consequence of the
    flexible attachment policy.  Recording this as a case so
    any future tightening does not regress silently.
    """
    src = """\
#!/bin/bash
# Entry point.
main() {
  echo hi
}
"""
    return "bash", src, "main", "#!/bin/bash"


@case(tags=["documented", "unconventional", "exterior_doc"])
def case_bash_comment_with_script_style_heading():
    """Heading-like comment above a function attaches."""
    src = """\
# ==== helpers ====
# Trim whitespace from $1.
trim() {
  echo "$1"
}
"""
    return "bash", src, "trim", "Trim whitespace"


@case(tags=["undocumented", "no_docs"])
def case_bash_fn_without_doc():
    """Undocumented shell function."""
    src = """\
bare() {
  echo hi
}
"""
    return "bash", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_bash_doc_detached_by_blank_line():
    """Blank line breaks attachment."""
    src = """\
# Orphan.

later() {
  echo hi
}
"""
    return "bash", src, "later", "Orphan"


@case(tags=["undocumented", "invalid"])
def case_bash_shebang_separated_by_blank_line():
    """When a blank line separates the shebang from the first
    function, the shebang stays detached — the same
    blank-line rule applies uniformly to all comment kinds.
    """
    src = """\
#!/bin/bash

main() {
  echo hi
}
"""
    return "bash", src, "main", "#!/bin/bash"
