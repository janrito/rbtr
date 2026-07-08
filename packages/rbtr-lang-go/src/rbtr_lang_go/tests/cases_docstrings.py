"""Go docstring-extraction test cases.

Each `@case` returns `(lang, source, symbol_name, snippet)` consumed by `test_docstrings.py`.
"""

from __future__ import annotations

from pytest_cases import case

type DocstringCase = tuple[str, str, str, str]


@case(tags=["documented", "canonical", "exterior_doc"])
def case_go_doc_comment_on_function() -> DocstringCase:
    """Canonical Go doc comment above `func`."""
    src = """\
package main

// Greet says hello.
func Greet() {}
"""
    return "go", src, "Greet", "Greet says hello"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_go_doc_comment_on_type() -> DocstringCase:
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
def case_go_doc_comment_on_method() -> DocstringCase:
    """Doc comment above a method receiver."""
    src = """\
package main

type T struct{}

// Do performs the action.
func (t *T) Do() {}
"""
    return "go", src, "Do", "Do performs the action"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_go_multi_line_doc_comment() -> DocstringCase:
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
def case_go_block_comment() -> DocstringCase:
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
def case_go_non_godoc_style_comment() -> DocstringCase:
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
def case_go_fn_without_doc() -> DocstringCase:
    """Undocumented function."""
    src = """\
package main

func bare() {}
"""
    return "go", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_go_doc_detached_by_blank_line() -> DocstringCase:
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
def case_go_doc_comment_above_previous_function_not_attached() -> DocstringCase:
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
