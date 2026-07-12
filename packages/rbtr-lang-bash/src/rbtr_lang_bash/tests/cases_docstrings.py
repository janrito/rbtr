"""Bash docstring-extraction test cases.

tree-sitter-bash uses a single `comment` node for any `#` line. Function
docs are a `#` comment run directly above the definition; a shebang
(`#!/bin/bash`) attaches to the first function only when no blank line
separates them.

Each `@case` returns `(lang, source, symbol_name, snippet)`; see
`test_docstrings.py` for the assertion direction per tag.
"""

from __future__ import annotations

from pytest_cases import case

type DocstringCase = tuple[str, str, str, str]


@case(tags=["documented", "canonical", "exterior_doc"])
def case_bash_hash_doc_on_function() -> DocstringCase:
    """Canonical `#` comment above a shell function."""
    src = """\
# Greet the user.
greet() {
  echo hello
}
"""
    return "bash", src, "greet", "Greet the user"


@case(tags=["documented", "canonical", "exterior_doc"])
def case_bash_multi_line_hash_doc() -> DocstringCase:
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
def case_bash_shebang_attached_to_first_function() -> DocstringCase:
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
def case_bash_comment_with_script_style_heading() -> DocstringCase:
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
def case_bash_fn_without_doc() -> DocstringCase:
    """Undocumented shell function."""
    src = """\
bare() {
  echo hi
}
"""
    return "bash", src, "bare", "PHANTOM_DOC_TEXT_SHOULD_NEVER_APPEAR"


@case(tags=["undocumented", "boundary_not_attached"])
def case_bash_doc_detached_by_blank_line() -> DocstringCase:
    """Blank line breaks attachment."""
    src = """\
# Orphan.

later() {
  echo hi
}
"""
    return "bash", src, "later", "Orphan"


@case(tags=["undocumented", "invalid"])
def case_bash_shebang_separated_by_blank_line() -> DocstringCase:
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
