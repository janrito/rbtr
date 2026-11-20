"""C docstring-extraction test cases.

Each `@case` returns `(files, symbol_name, snippet)` consumed by `test_docstrings.py`.
"""

from __future__ import annotations

from pytest_cases import case

type DocstringCase = tuple[dict[str, str], str, str]


@case(tags=["documented"])
def case_c_doxygen_on_function() -> DocstringCase:
    """Canonical Doxygen `/** */` above a function."""
    src = """\
/** Compute the sum. */
int add(int a, int b) { return a + b; }
"""
    return {"input.c": src}, "add", "Compute the sum"


@case(tags=["documented"])
def case_c_doxygen_on_struct() -> DocstringCase:
    """Doxygen above a struct."""
    src = """\
/** Point in 2D space. */
struct Point { int x; int y; };
"""
    return {"input.c": src}, "Point", "Point in 2D space"


@case(tags=["documented"])
def case_c_multi_line_doxygen() -> DocstringCase:
    r"""Multi-line Doxygen with `\param` / `\return` tags."""
    src = """\
/**
 * Hash a buffer.
 * \\param data the buffer
 * \\return the hash
 */
int hash(const char *data) { return 0; }
"""
    return {"input.c": src}, "hash", r"\return the hash"


@case(tags=["documented"])
def case_c_line_comment_run() -> DocstringCase:
    """Plain `//` comment run — common in embedded code where
    Doxygen style is heavier than needed.
    """
    src = """\
// Simple comment.
// Second line.
int foo(void) { return 0; }
"""
    return {"input.c": src}, "foo", "Simple comment"


@case(tags=["undocumented"])
def case_c_fn_without_doc() -> DocstringCase:
    src = """\
int bare(void) { return 0; }
"""
    return {"input.c": src}, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented"])
def case_c_doc_detached_by_blank_line() -> DocstringCase:
    """Blank line breaks attachment."""
    src = """\
/** Orphan. */

int later(void) { return 0; }
"""
    return {"input.c": src}, "later", "Orphan"


@case(tags=["undocumented"])
def case_c_doc_between_two_functions() -> DocstringCase:
    """Comment between two functions belongs to the later one."""
    src = """\
int first(void) { return 0; }

/** Doc for second. */
int second(void) { return 0; }
"""
    return {"input.c": src}, "first", "Doc for second"
