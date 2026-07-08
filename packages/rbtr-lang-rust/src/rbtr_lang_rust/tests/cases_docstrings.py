"""Rust docstring-extraction test cases.

Each `@case` returns `(lang, source, symbol_name, snippet)` consumed by `test_docstrings.py`.
"""

from __future__ import annotations

from pytest_cases import case

type DocstringCase = tuple[str, str, str, str]


@case(tags=["documented", "canonical", "exterior_doc"])
def case_rust_triple_slash_on_fn() -> DocstringCase:
    """Canonical `///` doc comment above a function."""
    src = """\
/// Greet the user.
fn greet() {}
"""
    return "rust", src, "greet", "Greet the user"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_rust_triple_slash_on_struct() -> DocstringCase:
    """`///` above a struct declaration."""
    src = """\
/// A widget.
struct Widget;
"""
    return "rust", src, "Widget", "A widget"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_rust_triple_slash_on_enum() -> DocstringCase:
    """`///` above an enum declaration."""
    src = """\
/// Colours.
enum Colour { Red, Green }
"""
    return "rust", src, "Colour", "Colours"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_rust_multi_line_triple_slash() -> DocstringCase:
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
def case_rust_block_doc_comment() -> DocstringCase:
    """`/** */` block doc comment — parsed as `block_comment`
    in the grammar.
    """
    src = """\
/** Block doc. */
fn foo() {}
"""
    return "rust", src, "foo", "Block doc"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_rust_impl_block_doc() -> DocstringCase:
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
def case_rust_plain_line_comment() -> DocstringCase:
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
def case_rust_fn_without_docs() -> DocstringCase:
    """Undocumented function."""
    src = """\
fn bare() {}
"""
    return "rust", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_rust_doc_detached_by_blank_line() -> DocstringCase:
    """A blank line between the doc comment and the function
    breaks attachment.  This is the tree-sitter-rust trailing-
    newline edge case: the line_comment span includes its own
    `\\n`, so "blank line" is detected via a `>= 2` newline
    count over `[content_end_without_nl, next_start)`.
    """
    src = """\
/// Orphaned comment.

fn later() {}
"""
    return "rust", src, "later", "Orphaned comment"


@case(tags=["undocumented", "invalid"])
def case_rust_inner_doc_not_attached() -> DocstringCase:
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
