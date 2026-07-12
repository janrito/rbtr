"""C++ docstring-extraction test cases.

Each `@case` returns `(lang, source, symbol_name, snippet)` consumed by `test_docstrings.py`.
"""

from __future__ import annotations

from pytest_cases import case

type DocstringCase = tuple[str, str, str, str]


@case(tags=["documented", "canonical", "exterior_doc"])
def case_cpp_doxygen_on_class() -> DocstringCase:
    """Doxygen above a C++ class."""
    src = """\
/** A widget. */
class Widget { public: int x; };
"""
    return "cpp", src, "Widget", "A widget"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_cpp_doxygen_on_function() -> DocstringCase:
    """Doxygen above a C++ function."""
    src = """\
/** Get the answer. */
int answer() { return 42; }
"""
    return "cpp", src, "answer", "Get the answer"


@case(tags=["documented", "edge_case", "exterior_doc"])
def case_cpp_triple_slash_style() -> DocstringCase:
    """Doxygen `///` style (single-line convention)."""
    src = """\
/// Triple-slash style.
int foo() { return 0; }
"""
    return "cpp", src, "foo", "Triple-slash style"


@case(tags=["undocumented", "no_docs"])
def case_cpp_fn_without_doc() -> DocstringCase:
    src = """\
int bare() { return 0; }
"""
    return "cpp", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_cpp_doc_detached_by_blank_line() -> DocstringCase:
    src = """\
/** Orphan. */

int later() { return 0; }
"""
    return "cpp", src, "later", "Orphan"
